//! Implementations of simple integration algorithms

/// Perform a 1D numerical integration on some arbitrary function over the
/// specified range using the [*adaptive Simpson
/// method*](https://en.wikipedia.org/wiki/Adaptive_Simpson's_method).
///
/// Unlike standard quadrature, this algorithm requires an error tolerance and
/// maximum recursion depth as inputs and the number of steps is adapter based
/// on these constraints.  The tolerance should, approximately, be the error in
/// the integral; however, the error can only ever be estimated.
///
/// In order to avoid infinite loops, a maximum recursion depths is set so that
/// the function exits even if the estimated error is greater than the target
/// error.  Additionally, a minimum recursion depth can be specified if the
/// function is known to be troublesome.  Be aware that computation time grows
/// exponentially with recursion depth!
///
/// A pointer to the function is expected, as a result, closures require an `&`
/// beforehand as in the example below.
///
/// # Example
///
/// ```
/// use ctlab::integration::simpson_adaptive;
///
/// let integral = simpson_adaptive(&|x| x.powf(x.sin()), (0., 10.), 1e-7, 3, 10);
/// let expect = 19.6371739097294805359245112109;
/// let diff = ((integral - expect) / (integral + expect)).abs();
/// assert!(diff < 1e-7);
/// ```
///
/// # Warning
///
/// Adaptive algorithms will in general use less points in regions where the
/// input function is flat, and more points where the function changes more
/// rapidly; however, it determines whether more points are needed based on the
/// midpoint *only*.  If the start, mid and end points all happen to be similar
/// (for example in a oscillatory function), the algorithm will terminate
/// unexpectedly early.
///
/// Adaptive algorithms can also struggle with discontinuous functions,
/// especially if the function changes significantly only over a small interval
/// as the algorithm may not 'see' the discontinuity at all.
///
/// Setting a high value of `min_recurse` may help some of these, at the cost of
/// possibly introducing unnecessary computations.
pub fn simpson_adaptive<F>(
    f: &F,
    (start, end): (f64, f64),
    tolerance: f64,
    min_recurse: usize,
    max_recurse: usize,
) -> f64
where
    F: Fn(f64) -> f64,
{
    debug_assert!(
        min_recurse <= max_recurse,
        "Minimum recursion depth cannot be greater than the maximum recursion depth."
    );

    let interval_size = end - start;
    let midpoint = (start + end) / 2.;

    let fs = f(start);
    let fm = f(midpoint);
    let fe = f(end);

    let sum = (interval_size / 6.0) * (fs + 4.0 * fm + fe);

    simpson_adaptive_aux(
        f,
        (start, end),
        tolerance,
        min_recurse,
        max_recurse,
        sum,
        fs,
        fm,
        fe,
    )
}

/// The auxillary function used in the adaptive Simpson method.  This uses
/// pre-computed results (hence the fairly large number of arguments) and
/// calculates the integral over the left and right half of the given interval.
/// If the new sums differ 'significantly' from the initial sum estimate, the
/// intervals are further subdivided.
#[allow(clippy::too_many_arguments)]
fn simpson_adaptive_aux<F>(
    f: &F,
    (start, end): (f64, f64),
    tolerance: f64,
    min_recurse: usize,
    max_recurse: usize,
    sum: f64,
    fs: f64,
    fm: f64,
    fe: f64,
) -> f64
where
    F: Fn(f64) -> f64,
{
    let midpoint = (start + end) / 2.;
    let interval_size = end - start;

    let left_midpoint = (start + midpoint) / 2.;
    let right_midpoint = (midpoint + end) / 2.;

    let flm = f(left_midpoint);
    let frm = f(right_midpoint);

    let sum_left = (interval_size / 12.0) * (fs + 4. * flm + fm);
    let sum_right = (interval_size / 12.0) * (fm + 4. * frm + fe);
    let sum2 = sum_left + sum_right;

    if min_recurse > 0 {
        simpson_adaptive_aux(
            f,
            (start, midpoint),
            tolerance / 2.,
            min_recurse - 1,
            max_recurse - 1,
            sum_left,
            fs,
            flm,
            fm,
        ) + simpson_adaptive_aux(
            f,
            (midpoint, end),
            tolerance / 2.,
            min_recurse - 1,
            max_recurse - 1,
            sum_right,
            fm,
            frm,
            fe,
        )
    } else if max_recurse == 0 || (sum2 - sum).abs() < 15. * tolerance {
        sum2 + (sum2 - sum) / 15.
    } else {
        simpson_adaptive_aux(
            f,
            (start, midpoint),
            tolerance / 2.,
            0,
            max_recurse - 1,
            sum_left,
            fs,
            flm,
            fm,
        ) + simpson_adaptive_aux(
            f,
            (midpoint, end),
            tolerance / 2.,
            0,
            max_recurse - 1,
            sum_right,
            fm,
            frm,
            fe,
        )
    }
}
