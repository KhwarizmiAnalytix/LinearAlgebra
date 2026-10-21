#include <cstdint>
#include <limits>
#include <string>

#include "test_reporting.h"

namespace
{
enum class signed_code : int
{
    failure = -7
};

enum class unsigned_code : std::uint64_t
{
    largest = std::numeric_limits<std::uint64_t>::max()
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
