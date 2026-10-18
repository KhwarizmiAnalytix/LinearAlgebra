#include "include/matrix_operation/trace.h"

int main()
{
    const double matrix[] = {1.0, 2.0, 3.0, 4.0};
    const double result =
        linalg::trace<double, linalg::linalg_backend::native>::apply(matrix, 2, 2);
    return result == 5.0 ? 0 : 1;
}
