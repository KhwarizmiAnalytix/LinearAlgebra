#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "test_reporting.h"

namespace
{
// NOLINTNEXTLINE(performance-enum-size): preserve the vendor API enum representation in this runtime test double.
enum class signed_code : int
{
    failure = -7
};

enum class unsigned_code : std::uint64_t
{
    largest = std::numeric_limits<std::uint64_t>::max()
};

struct captured_reports
{
    explicit captured_reports(logging::logger_verbosity_enum verbosity)
    {
        logging::logger::add_callback(
            "linalg-reporting-test",
            [](void* data, const logging::logger::Message& message)
            { static_cast<std::vector<logging::logger::Message>*>(data)->push_back(message); },
            &messages_,
            verbosity);
    }

    ~captured_reports() { logging::logger::remove_callback("linalg-reporting-test"); }

    std::vector<logging::logger::Message> messages_;
};
}  // namespace

TEST(Reporting, FormatsEnums)
{
    using linalg_test::reporting::value_string;
    EXPECT_EQ(value_string(signed_code::failure), "-7");
    EXPECT_EQ(value_string(unsigned_code::largest), "18446744073709551615");
}

TEST(Reporting, FormatsNonEnums)
{
    using linalg_test::reporting::value_string;
    EXPECT_EQ(value_string(true), "true");
    EXPECT_EQ(value_string(false), "false");
    EXPECT_EQ(value_string(42), "42");
    EXPECT_EQ(value_string(1.5f), "1.5");
    EXPECT_EQ(value_string(2.5), "2.5");
    EXPECT_EQ(value_string(std::string("text")), "text");
}

TEST(Reporting, SuccessfulChecksDoNotLogAtInfoLevel)
{
    captured_reports messages(logging::logger_verbosity_enum::VERBOSITY_INFO);
    const auto       result = linalg_test::reporting::equal("actual", "expected", 42, 42);
    EXPECT_TRUE(static_cast<bool>(result));
    EXPECT_TRUE(messages.messages_.empty());
}

TEST(Reporting, SuccessfulChecksRemainAvailableAtTraceLevel)
{
    captured_reports messages(logging::logger_verbosity_enum::VERBOSITY_TRACE);
    const auto       result = linalg_test::reporting::equal("actual", "expected", 42, 42);
    const auto       logged = messages.messages_;
    EXPECT_TRUE(static_cast<bool>(result));
    ASSERT_EQ(logged.size(), 1U);
    EXPECT_EQ(logged[0].verbosity, logging::logger_verbosity_enum::VERBOSITY_TRACE);
    EXPECT_NE(logged[0].message.find("[PASS]"), std::string::npos);
    EXPECT_NE(logged[0].message.find("expected=42, actual=42, difference=0"), std::string::npos);
}

TEST(Reporting, FailedChecksPreserveDiagnostics)
{
    captured_reports messages(logging::logger_verbosity_enum::VERBOSITY_INFO);
    const auto       result = linalg_test::reporting::equal("actual", "expected", 41, 42);
    const auto       logged = messages.messages_;
    EXPECT_FALSE(result);
    ASSERT_EQ(logged.size(), 1U);
    EXPECT_EQ(logged[0].verbosity, logging::logger_verbosity_enum::VERBOSITY_ERROR);
    EXPECT_NE(logged[0].message.find("[FAIL]"), std::string::npos);
    EXPECT_NE(std::string(result.message()).find("actual == expected"), std::string::npos);
    EXPECT_NE(std::string(result.message()).find("expected=42, actual=41, difference=1"),
        std::string::npos);
}
