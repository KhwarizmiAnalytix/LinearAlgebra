#pragma once

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include "include/logging.h"
#include "gtest/gtest.h"

namespace linalg_test
{
namespace reporting
{

template <typename T, typename = void> struct is_streamable : std::false_type
{
};

template <typename T>
struct is_streamable<T,
    std::void_t<decltype(std::declval<std::ostream&>() << std::declval<const T&>())>>
    : std::true_type
{
};

template <typename T> std::string value_string(const T& value)
{
    using value_type = typename std::decay<T>::type;
    if constexpr (std::is_same<value_type, bool>::value)
    {
        return value ? "true" : "false";
    }
    else if constexpr (std::is_enum<value_type>::value)
    {
        return std::to_string(static_cast<typename std::underlying_type<value_type>::type>(value));
    }
    else if constexpr (std::is_pointer<value_type>::value)
    {
        std::ostringstream output;
        output << static_cast<const void*>(value);
        return output.str();
    }
    else if constexpr (std::is_arithmetic<value_type>::value)
    {
        std::ostringstream output;
        output << std::setprecision(std::numeric_limits<value_type>::max_digits10) << value;
        return output.str();
    }
    else if constexpr (is_streamable<value_type>::value)
    {
        std::ostringstream output;
        output << value;
        return output.str();
    }
    else
    {
        return "<value>";
    }
}

template <typename Left, typename Right>
std::string difference_string(const Left& actual, const Right& expected)
{
    using left_type  = typename std::decay<Left>::type;
    using right_type = typename std::decay<Right>::type;
    if constexpr ((std::is_arithmetic<left_type>::value || std::is_enum<left_type>::value) &&
                  (std::is_arithmetic<right_type>::value || std::is_enum<right_type>::value))
    {
        const long double difference =
            std::fabs(static_cast<long double>(actual) - static_cast<long double>(expected));
        std::ostringstream output;
        output << std::setprecision(std::numeric_limits<long double>::max_digits10) << difference;
        return output.str();
    }
    else
    {
        return "n/a";
    }
}

inline std::string current_test_name()
{
    const auto* test = ::testing::UnitTest::GetInstance()->current_test_info();
    return test ? std::string(test->test_suite_name()) + "." + test->name() : "<no active test>";
}

inline ::testing::AssertionResult report(const char* check,
    bool                                             passed,
    const std::string&                               expected,
    const std::string&                               actual,
    const std::string&                               difference,
    const std::string&                               tolerance = "n/a",
    const std::string&                               condition = "n/a")
{
    const std::string message =
        std::string("method tested/check=") + check + "\ndetails: test=" + current_test_name() +
        "\nexpected=" + expected + ", actual=" + actual + ", difference=" + difference +
        ", tolerance=" + tolerance + ", condition=" + condition;
    if (passed)
    {
        LOGGING_LOG_INFO("[PASS] {}\n{}", check, message);
        return ::testing::AssertionSuccess();
    }

    LOGGING_LOG_ERROR("[FAIL] {}\n{}", check, message);
    return ::testing::AssertionFailure() << message;
}

template <typename Left, typename Right>
::testing::AssertionResult equal(const char* left_expression,
    const char*                              right_expression,
    const Left&                              actual,
    const Right&                             expected)
{
    return report((std::string(left_expression) + " == " + right_expression).c_str(),
        actual == expected,
        value_string(expected),
        value_string(actual),
        difference_string(actual, expected),
        "n/a",
        "actual == expected");
}

template <typename Left, typename Right>
::testing::AssertionResult less(const char* left_expression,
    const char*                             right_expression,
    const Left&                             actual,
    const Right&                            expected)
{
    return report((std::string(left_expression) + " < " + right_expression).c_str(),
        actual < expected,
        value_string(expected),
        value_string(actual),
        difference_string(actual, expected),
        "n/a",
        "actual < expected");
}

template <typename Left, typename Right>
::testing::AssertionResult less_equal(const char* left_expression,
    const char*                                   right_expression,
    const Left&                                   actual,
    const Right&                                  expected)
{
    return report((std::string(left_expression) + " <= " + right_expression).c_str(),
        actual <= expected,
        value_string(expected),
        value_string(actual),
        difference_string(actual, expected),
        "n/a",
        "actual <= expected");
}

template <typename Left, typename Right>
::testing::AssertionResult greater(const char* left_expression,
    const char*                                right_expression,
    const Left&                                actual,
    const Right&                               expected)
{
    return report((std::string(left_expression) + " > " + right_expression).c_str(),
        actual > expected,
        value_string(expected),
        value_string(actual),
        difference_string(actual, expected),
        "n/a",
        "actual > expected");
}

template <typename Actual, typename Expected, typename Tolerance>
::testing::AssertionResult near(const char* actual_expression,
    const char*                             expected_expression,
    const char*                             tolerance_expression,
    const Actual&                           actual,
    const Expected&                         expected,
    const Tolerance&                        tolerance)
{
    const long double difference =
        std::fabs(static_cast<long double>(actual) - static_cast<long double>(expected));
    std::ostringstream difference_output;
    difference_output << std::setprecision(std::numeric_limits<long double>::max_digits10)
                      << difference;
    return report((std::string(actual_expression) + " ~= " + expected_expression).c_str(),
        difference <= static_cast<long double>(tolerance),
        value_string(expected),
        value_string(actual),
        difference_output.str(),
        std::string(tolerance_expression) + "=" + value_string(tolerance),
        "absolute difference <= tolerance");
}

template <bool Expected>::testing::AssertionResult boolean(const char* expression, bool actual)
{
    return report(expression,
        actual == Expected,
        Expected ? "true" : "false",
        value_string(actual),
        difference_string(actual, Expected));
}

template <typename Callable>
::testing::AssertionResult any_throw(
    const char*, const char*, const char* statement, const Callable& call)
{
    bool threw = false;
    try
    {
        call();
    }
    catch (...)
    {
        threw = true;
    }
    return report(statement, threw, "exception", threw ? "exception" : "no exception", "n/a");
}

template <typename Exception, typename Callable>
::testing::AssertionResult throws(const char*,
    const char*,
    const char*,
    const char*     statement,
    const char*     expected_type,
    const Callable& call)
{
    bool caught_expected = false;
    bool caught_other    = false;
    try
    {
        call();
    }
    catch (const Exception&)
    {
        caught_expected = true;
    }
    catch (...)
    {
        caught_other = true;
    }

    const std::string expected = std::string("exception ") + expected_type;
    const std::string actual =
        caught_expected ? expected : (caught_other ? "other exception" : "no exception");
    return report(statement, caught_expected, expected, actual, "n/a");
}

template <typename Callable>
::testing::AssertionResult no_throw(
    const char*, const char*, const char* statement, const Callable& call)
{
    bool threw = false;
    try
    {
        call();
    }
    catch (...)
    {
        threw = true;
    }
    return report(statement, !threw, "no exception", threw ? "exception" : "no exception", "n/a");
}

}  // namespace reporting
}  // namespace linalg_test

#undef EXPECT_EQ
#define EXPECT_EQ(actual, expected)                                                                \
    EXPECT_PRED_FORMAT2(::linalg_test::reporting::equal, actual, expected)
#undef ASSERT_EQ
#define ASSERT_EQ(actual, expected)                                                                \
    ASSERT_PRED_FORMAT2(::linalg_test::reporting::equal, actual, expected)
#undef EXPECT_LT
#define EXPECT_LT(actual, expected)                                                                \
    EXPECT_PRED_FORMAT2(::linalg_test::reporting::less, actual, expected)
#undef EXPECT_LE
#define EXPECT_LE(actual, expected)                                                                \
    EXPECT_PRED_FORMAT2(::linalg_test::reporting::less_equal, actual, expected)
#undef EXPECT_GT
#define EXPECT_GT(actual, expected)                                                                \
    EXPECT_PRED_FORMAT2(::linalg_test::reporting::greater, actual, expected)
#undef EXPECT_NEAR
#define EXPECT_NEAR(actual, expected, tolerance)                                                   \
    EXPECT_PRED_FORMAT3(::linalg_test::reporting::near, actual, expected, tolerance)
#undef EXPECT_TRUE
#define EXPECT_TRUE(actual) EXPECT_PRED_FORMAT1(::linalg_test::reporting::boolean<true>, actual)
#undef EXPECT_ANY_THROW
#define EXPECT_ANY_THROW(statement)                                                                \
    EXPECT_PRED_FORMAT2(::linalg_test::reporting::any_throw, #statement, [&]() { statement; })
#undef EXPECT_THROW
#define EXPECT_THROW(statement, exception_type)                                                    \
    EXPECT_PRED_FORMAT3(::linalg_test::reporting::throws<exception_type>,                          \
        #statement,                                                                                \
        #exception_type,                                                                           \
        [&]() { statement; })
#undef EXPECT_NO_THROW
#define EXPECT_NO_THROW(statement)                                                                 \
    EXPECT_PRED_FORMAT2(::linalg_test::reporting::no_throw, #statement, [&]() { statement; })
