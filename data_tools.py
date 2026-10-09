# =============================================================================
# data_tools.py  -  Macro variable selection engine
# =============================================================================
# GOAL OF THIS FILE
#   The user gives us a list of FRED series codes and picks ONE of them as the
#   TARGET (the thing to be explained, e.g. industrial production). We then
#   work out which of the OTHER variables matter most for explaining the
#   target, and check whether a linear regression using them is trustworthy.
#
# THE PIPELINE (each step is one function below, run in this order)
#   1. get_clean_table        download from FRED, make monthly, align, fill small holes
#   2. make_stationary        ADF + KPSS tests; fix "wandering" series
#   3. single_variable_ranking  how well does each variable explain the target alone?
#   4. forward_selection      add variables one by one while ADJUSTED R2 improves
#   5. fit_final_model        final regression + robust (Newey-West) significance
#   6. compute_vif            multicollinearity check
#   7. run_diagnostics        autocorrelation, heteroskedasticity, normality tests
#   8. run_analysis           runs all of the above and returns ONE dictionary
#                             (the website / chatbot will use that dictionary)
#
# WORDS USED IN THE COMMENTS
#   target (y)      the variable we want to explain
#   predictors (X)  the variables we use to explain it
#   residuals       what the model gets wrong: actual value minus model's value
#   p-value         a number from 0 to 1. For a test with a "null hypothesis"
#                   (the boring default claim), the p-value answers: "if the
#                   boring claim were true, how surprising is my data?"
#                   Small p (below 0.05) = very surprising = reject the boring claim.
#   n               number of observations (months)
#   k               number of predictors
# =============================================================================

import os
import warnings

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan

pd.set_option("display.width", 200)

# statsmodels prints a long "FutureWarning" about how results will be returned
# in a future version. It is harmless, so we hide just that message.
warnings.filterwarnings("ignore", message=".*returns a plain tuple.*",
                        category=FutureWarning)

# Read the FRED key from the .env file (so the key never sits inside the code).
load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

SIGNIFICANCE = 0.05   # the usual cut-off for "statistically significant"


# =============================================================================
# STEP 1: DOWNLOAD AND CLEAN THE DATA
# =============================================================================
def get_clean_table(codes, max_fill=2):
    """
    Download each series, convert to ONE value per month, keep only the period
    where every series exists, and fill SMALL holes.

    Why monthly?   Regression needs all variables on the same dates. Daily,
                   monthly and quarterly data can't be lined up directly, so
                   we convert everything to monthly (average of the month).
    Why trim?      A series that starts in 1990 can't help before 1990, so the
                   table starts at the LATEST first-date and ends at the
                   EARLIEST last-date. The shortest series sets the window.
    Why fill?      A missing month (e.g. Oct 2025, US government shutdown)
                   would break the 'one row = one month' rule. For a short
                   gap we draw a straight line between the month before and
                   the month after (for a 1-month gap this is exactly their
                   average). We do NOT use the median: for trending data
                   (CPI went 26 -> 334) the median would create a fake crash.
    Long gaps     (more than max_fill months in a row) are NOT invented; those
                   months are removed and a warning is added.

    Returns: (table, notes)  notes = plain-English messages for the user.
    """
    notes = []
    columns = {}

    for code in codes:
        raw = fred.get_series(code).dropna()

        # Detect quarterly / annual series: the typical spacing between
        # observations is much longer than a month. We don't support those
        # yet, so we skip them and tell the user.
        spacing_days = raw.index.to_series().diff().dt.days.median()
        if spacing_days > 45:
            notes.append(f"{code}: skipped. Data is quarterly/annual, only "
                         f"monthly or daily series are supported.")
            continue

        # "MS" = month start. .mean() averages several readings in a month
        # (e.g. daily data); for monthly data it simply keeps the value.
        columns[code] = raw.resample("MS").mean()

    if len(columns) == 0:
        raise ValueError("None of the requested series could be used.")

    table = pd.DataFrame(columns)   # lines everything up by date

    # Keep only the overlap period (see 'Why trim?' above).
    start = table.apply(lambda c: c.first_valid_index()).max()
    end = table.apply(lambda c: c.last_valid_index()).min()
    table = table.loc[start:end]

    # Fill interior gaps with a straight line between neighbours.
    filled = table.interpolate(limit_area="inside")

    # Work out which gaps are too long to trust.
    too_long_rows = pd.Series(False, index=table.index)
    for code in table.columns:
        col = table[code]
        # length of the run of blanks each blank belongs to
        run_length = col.isna().groupby(col.notna().cumsum()).transform("sum")
        long_gap = col.isna() & (run_length > max_fill)
        too_long_rows |= long_gap

        for month in table.index[col.isna()]:
            if long_gap[month]:
                notes.append(f"{code}: {month:%b %Y} was missing in a gap longer "
                             f"than {max_fill} months; the month was removed "
                             f"(WARNING: months are no longer evenly spaced).")
            else:
                notes.append(f"{code}: {month:%b %Y} was missing and was replaced "
                             f"with {filled.loc[month, code]:.2f} (straight line "
                             f"between the months before and after).")

    table = filled.loc[~too_long_rows]
    return table, notes


# =============================================================================
# STEP 2: STATIONARITY
# =============================================================================
# WHY THIS MATTERS
#   A STATIONARY series keeps returning to a steady average (like a dog on a
#   leash). A NON-STATIONARY series wanders off (like a person walking with no
#   destination): CPI, GDP and industrial production all trend upward for
#   decades.
#   If you regress one wandering series on another, you often get a HIGH R2
#   and "significant" results even when they are totally unrelated. This is
#   called a SPURIOUS REGRESSION, the biggest trap in macro analysis. So we
#   test every variable first and convert the wandering ones.
#
# THE TWO TESTS (we use both, because each has a blind spot)
#   ADF  (Augmented Dickey-Fuller)
#        Null hypothesis (boring claim): the series has a unit root, i.e. it WANDERS.
#        Formula idea: regress today's CHANGE on yesterday's LEVEL (plus some
#        lagged changes). If the level has no pull-back effect, it wanders.
#        p < 0.05  -> reject the claim -> series is STATIONARY (good).
#   KPSS (Kwiatkowski-Phillips-Schmidt-Shin)
#        Null hypothesis is the OPPOSITE: the series IS stationary.
#        p < 0.05  -> reject -> series is NON-stationary.
#        (statsmodels only reports KPSS p-values between 0.01 and 0.10.)
#   Our rule: call it stationary only if ADF says stationary AND KPSS agrees.
#   If they disagree, we play safe and transform it.
#
# THE FIX WHEN IT WANDERS
#   - Series that grew a lot (largest value > 3x smallest, all positive, like
#     CPI or industrial production): LOG-DIFFERENCE = log(today) - log(last month).
#     This is the monthly GROWTH RATE (e.g. CPI -> inflation). Percentage
#     changes stay comparable over decades; plain point changes would not
#     (a 1-point CPI move meant a lot in 1954 and very little in 2026).
#   - Other series (rates like unemployment, Fed funds): simple DIFFERENCE =
#     today - last month.
#   After transforming we test again and show the new p-values.
#
# HOW TO READ THE RESULT
#   "stationary_after = True" for all variables means we can regress safely.
#   If a variable is still non-stationary after one transformation, the app
#   warns the user (it may need a second difference or a cointegration test).
#   NOTE: after this step the variables mean "monthly change / growth", so the
#   regression answers "which variable's CHANGES explain the target's CHANGES".
# =============================================================================
def stationarity_check(series):
    """Run ADF and KPSS on one series. Returns (adf_p, kpss_p, is_stationary)."""
    s = series.dropna()
    adf_p = adfuller(s)[1]                       # index 1 of the result = p-value
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # KPSS prints harmless warnings
        kpss_p = kpss(s, regression="c", nlags="auto")[1]
    is_stationary = bool(adf_p < SIGNIFICANCE and kpss_p > SIGNIFICANCE)
    return float(adf_p), float(kpss_p), is_stationary


def _is_rate(code):
    """
    True if FRED says this series is measured in percent (interest rates,
    unemployment rate...). Rates must NEVER be log-differenced: a rate near
    zero (the Fed funds rate was ~0.1% in 2009-2015) makes log changes explode
    (0.07 -> 0.09 looks like a +25% 'growth'). Rates get a simple difference.
    If the lookup fails we assume it is not a rate.
    """
    try:
        return "Percent" in str(fred.get_series_info(code)["units"])
    except Exception:
        return False


def make_stationary(table):
    """
    Test every column; transform the ones that wander.
    Returns: (stationary_table, report)  report = one dictionary per variable.
    """
    converted = {}
    report = []

    for col in table.columns:
        s = table[col]
        adf_p, kpss_p, ok = stationarity_check(s)

        entry = {
            "variable": col,
            "adf_p": adf_p,
            "kpss_p": kpss_p,
            "stationary_before": ok,
            "transform": "none",
            "adf_p_after": None,
            "kpss_p_after": None,
            "stationary_after": ok,
        }

        if ok:
            converted[col] = s
            entry["message"] = (f"{col}: already stationary "
                                f"(ADF p={adf_p:.3f}, KPSS p={kpss_p:.3f}). Left as is.")
        else:
            if (not _is_rate(col)) and (s > 0).all() and s.max() / s.min() > 3:
                new = np.log(s).diff()
                entry["transform"] = "log-difference (monthly growth rate)"
            else:
                new = s.diff()
                entry["transform"] = "difference (monthly change)"

            adf_p2, kpss_p2, _ = stationarity_check(new)
            # AFTER transforming, ADF alone decides. KPSS often objects
            # mildly on very long samples that contain regime shifts (the
            # 1970s inflation, the 2020 crash). We show that as a caution,
            # not as a failure.
            ok2 = bool(adf_p2 < SIGNIFICANCE)
            converted[col] = new
            entry.update(adf_p_after=adf_p2, kpss_p_after=kpss_p2,
                         stationary_after=ok2)
            if not ok2:
                verdict = "STILL NOT stationary (use with caution)"
            elif kpss_p2 < SIGNIFICANCE:
                verdict = ("stationary by ADF (KPSS mildly disagrees; common in "
                           "long samples with regime shifts)")
            else:
                verdict = "stationary"
            entry["message"] = (
                f"{col}: NOT stationary (ADF p={adf_p:.3f}, KPSS p={kpss_p:.3f}). "
                f"Converted to {entry['transform']}. After: ADF p={adf_p2:.3f}, "
                f"KPSS p={kpss_p2:.3f} -> {verdict}.")

        report.append(entry)

    # The first row is blank after differencing, so drop it.
    stationary_table = pd.DataFrame(converted).dropna()
    return stationary_table, report


# =============================================================================
# STEP 3: HOW WELL DOES EACH VARIABLE EXPLAIN THE TARGET ON ITS OWN?
# =============================================================================
# REGRESSION IN ONE LINE
#   We fit:   target = a + b1*X1 + b2*X2 + ... + error
#   by OLS (Ordinary Least Squares): choose a, b1, b2... so that the sum of
#   squared errors is as small as possible.
#
# R2 (R-squared)
#   R2 = 1 - (sum of squared errors) / (total variation of the target)
#   "Share of the target's ups and downs explained by the model." 0 to 1.
#   With ONE predictor, R2 equals the squared correlation. With several, it
#   does not (that's why we use a proper regression).
#   PROBLEM: R2 can never go DOWN when you add a variable, even a useless one.
#
# ADJUSTED R2  (the score we use to decide what "matters")
#   Adj R2 = 1 - (1 - R2) * (n - 1) / (n - k - 1)
#   It subtracts a penalty for every extra variable. A new variable raises
#   Adj R2 only if it improves the fit by MORE than the penalty. If Adj R2
#   falls, the variable is not adding value.
#   Good to know: Adj R2 rises whenever the new variable's |t-statistic| > 1,
#   which is a fairly LENIENT bar, so we also check significance (Step 5).
# =============================================================================
def single_variable_ranking(table, target):
    """Fit target ~ one variable, for each variable. Sorted by Adj R2, best first."""
    y = table[target]
    rows = []
    for col in table.columns:
        if col == target:
            continue
        model = sm.OLS(y, sm.add_constant(table[[col]])).fit()
        rows.append({
            "variable": col,
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
            "coefficient": float(model.params[col]),   # sign shows direction of link
            "p_value": float(model.pvalues[col]),
        })
    rows.sort(key=lambda r: r["adj_r2"], reverse=True)
    return rows


# =============================================================================
# STEP 4: FORWARD SELECTION BY ADJUSTED R2
# =============================================================================
# HOW IT WORKS
#   Start with no variables (Adj R2 = 0).
#   Round 1: try each variable alone; keep the one with the highest Adj R2
#            (only if it beats the current score).
#   Round 2: try adding each remaining variable to the kept one(s); keep the
#            one that raises Adj R2 the most.
#   Stop when no remaining variable raises Adj R2.
#   The ORDER of entry is the ranking of "what matters most": the first one
#   is the most useful on its own, the next adds the most on top of it, etc.
#   Variables that never get in did not add value beyond what was selected.
#
# CAUTION: this measures statistical usefulness, NOT cause and effect.
# =============================================================================
def forward_selection(table, target):
    y = table[target]
    remaining = [c for c in table.columns if c != target]
    selected = []
    current_adj = 0.0
    steps = []

    while remaining:
        best_name, best_adj, best_r2 = None, current_adj, None
        for candidate in remaining:
            X = sm.add_constant(table[selected + [candidate]])
            model = sm.OLS(y, X).fit()
            if model.rsquared_adj > best_adj:
                best_name = candidate
                best_adj = float(model.rsquared_adj)
                best_r2 = float(model.rsquared)

        if best_name is None:        # nothing improves Adj R2 -> stop
            break

        steps.append({
            "step": len(steps) + 1,
            "added": best_name,
            "adj_r2": best_adj,
            "r2": best_r2,
            "gain_in_adj_r2": best_adj - current_adj,
        })
        selected.append(best_name)
        remaining.remove(best_name)
        current_adj = best_adj

    return selected, steps


# =============================================================================
# STEP 5: THE FINAL MODEL AND SIGNIFICANCE TESTS
# =============================================================================
# WHAT WE REPORT
#   R2, Adj R2        see Step 3.
#   F-test            Null: ALL coefficients are zero (model is useless).
#                     p < 0.05 -> the model as a whole explains something.
#   t-test (per var)  Null: this variable's coefficient is zero (no effect).
#                     t = coefficient / standard error. p < 0.05 -> significant.
#   AIC, BIC          Other scores that punish extra variables. LOWER is better.
#                     Use them to compare models (BIC punishes more strongly).
#
# NEWEY-WEST (HAC) STANDARD ERRORS
#   The normal t-test assumes the errors are not autocorrelated and have
#   constant spread. Macro data often breaks both (see Step 7). Newey-West
#   standard errors correct for that, so the p-values stay honest.
#   Coefficients themselves do not change, only their standard errors/p-values.
#   Number of lags used: floor(4 * (n/100)^(2/9)), a standard rule of thumb.
#   We show BOTH p-values; trust the "robust" one.
# =============================================================================
def fit_final_model(table, target, selected):
    y = table[target]
    X = sm.add_constant(table[selected])

    ols = sm.OLS(y, X).fit()                                   # classical
    n = int(ols.nobs)
    lags = int(np.floor(4 * (n / 100) ** (2 / 9)))             # Newey-West rule
    robust = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    summary = {
        "n_observations": n,
        "n_predictors": len(selected),
        "r2": float(ols.rsquared),
        "adj_r2": float(ols.rsquared_adj),
        "f_test_p_value": float(ols.f_pvalue),
        "aic": float(ols.aic),
        "bic": float(ols.bic),
        "newey_west_lags": lags,
    }

    coefficients = []
    for name in ols.params.index:
        coefficients.append({
            "variable": name,
            "coefficient": float(ols.params[name]),
            "p_value_classical": float(ols.pvalues[name]),
            "p_value_robust": float(robust.pvalues[name]),
            "significant_robust": bool(robust.pvalues[name] < SIGNIFICANCE),
        })
    return ols, summary, coefficients


# =============================================================================
# STEP 6: MULTICOLLINEARITY (VIF)
# =============================================================================
# PROBLEM
#   If two predictors carry almost the same information (e.g. two inflation
#   measures), the regression cannot tell which one deserves the credit. The
#   coefficients become unstable and their significance unreliable, even if
#   the overall R2 looks fine.
# TEST: VIF (Variance Inflation Factor)
#   For predictor j: regress it on all the OTHER predictors and get that R2 (R2_j).
#   VIF_j = 1 / (1 - R2_j)
#   VIF = 1  -> unrelated to the others (ideal)
#   VIF > 5  -> worth a look      VIF > 10 -> serious problem
#   FIX: drop one of the overlapping variables.
# =============================================================================
def compute_vif(table, selected):
    X = sm.add_constant(table[selected])
    rows = []
    for i, name in enumerate(X.columns):
        if name == "const":
            continue
        rows.append({
            "variable": name,
            "vif": float(variance_inflation_factor(X.values, i)),
        })
    return rows


# =============================================================================
# STEP 7: RESIDUAL DIAGNOSTICS
# =============================================================================
# The residuals (model errors) should look like random noise. If they have a
# pattern, the model is missing something or its significance tests are
# unreliable. Four checks:
#
# 1. DURBIN-WATSON  (autocorrelation, quick check)
#    DW = sum((e_t - e_{t-1})^2) / sum(e_t^2), always between 0 and 4.
#    About 2 = no autocorrelation. Below ~1.5 = errors follow each other
#    (positive autocorrelation, common in macro data). Above ~2.5 = negative.
#    No p-value; we use the 1.5 - 2.5 rule of thumb.
#
# 2. BREUSCH-GODFREY  (autocorrelation, formal test, 12 lags = one year)
#    Null: no autocorrelation up to 12 months.  p < 0.05 -> autocorrelation.
#    Why it matters: standard errors are too small, so variables look more
#    significant than they are. FIX: Newey-West errors (we already use them).
#
# 3. BREUSCH-PAGAN  (heteroskedasticity)
#    Null: the error spread is constant ("homoskedastic").
#    p < 0.05 -> spread changes with the predictors (e.g. bigger errors in
#    crises). Same consequence and same fix as above.
#
# 4. JARQUE-BERA  (normality of errors)
#    JB = n/6 * (skewness^2 + (kurtosis - 3)^2 / 4)
#    Null: errors are normally distributed.  p < 0.05 -> they are not.
#    Macro data has fat tails (crises), so with hundreds of months this test
#    almost always fails. The coefficients are still fine in large samples;
#    treat a failure as a mild caution, not a disaster.
# =============================================================================
def run_diagnostics(ols):
    resid = ols.resid
    results = []

    # --- Durbin-Watson ---
    dw = float(durbin_watson(resid))
    results.append({
        "test": "Durbin-Watson (autocorrelation)",
        "statistic": dw,
        "p_value": None,
        "passed": bool(1.5 <= dw <= 2.5),
        "meaning": "Values near 2 mean errors are not linked month to month.",
        "if_failed": "Errors are correlated over time; rely on Newey-West p-values.",
    })

    # --- Breusch-Godfrey ---
    _, bg_p, _, _ = acorr_breusch_godfrey(ols, nlags=12)
    results.append({
        "test": "Breusch-Godfrey (autocorrelation, 12 lags)",
        "statistic": None,
        "p_value": float(bg_p),
        "passed": bool(bg_p >= SIGNIFICANCE),
        "meaning": "Null: no autocorrelation. p >= 0.05 is good.",
        "if_failed": "Autocorrelation present; classical p-values are too optimistic. "
                     "Trust the robust (Newey-West) p-values.",
    })

    # --- Breusch-Pagan ---
    _, bp_p, _, _ = het_breuschpagan(resid, ols.model.exog)
    results.append({
        "test": "Breusch-Pagan (heteroskedasticity)",
        "statistic": None,
        "p_value": float(bp_p),
        "passed": bool(bp_p >= SIGNIFICANCE),
        "meaning": "Null: error spread is constant. p >= 0.05 is good.",
        "if_failed": "Error size changes over time/levels; classical p-values "
                     "unreliable. Trust the robust p-values.",
    })

    # --- Jarque-Bera ---
    jb, jb_p, skew, kurt = jarque_bera(resid)
    results.append({
        "test": "Jarque-Bera (normality of errors)",
        "statistic": float(jb),
        "p_value": float(jb_p),
        "passed": bool(jb_p >= SIGNIFICANCE),
        "meaning": f"Null: errors are normal. Skewness={skew:.2f}, kurtosis={kurt:.2f} "
                   f"(normal = 0 and 3).",
        "if_failed": "Errors have fat tails/skew (typical for macro data). Mild "
                     "caution only when the sample is large.",
    })
    return results


# =============================================================================
# STEP 8: RUN EVERYTHING AND RETURN ONE DICTIONARY
# =============================================================================
# The website will call this function and show the dictionary; the Groq AI
# will later be given this same dictionary and asked to explain it in words.
# =============================================================================
def run_analysis(codes, target):
    if target not in codes:
        raise ValueError("The target must be one of the selected series.")

    # Step 1
    raw_table, data_notes = get_clean_table(codes)
    if target not in raw_table.columns:
        raise ValueError("The target series could not be used (see data notes).")
    if raw_table.shape[1] < 2:
        raise ValueError("Need the target plus at least one more usable variable.")

    # Step 2
    table, stationarity = make_stationary(raw_table)
    if len(table) < 60:
        raise ValueError(f"Only {len(table)} months of data overlap; need at least 60.")

    flags = []   # plain-English warnings collected from all steps

    for s in stationarity:
        if not s["stationary_after"]:
            flags.append(f"{s['variable']} is still non-stationary after "
                         f"transformation; results involving it may be spurious.")

    # Steps 3 and 4
    ranking = single_variable_ranking(table, target)
    selected, steps = forward_selection(table, target)

    result = {
        "target": target,
        "period": {"start": f"{table.index[0]:%Y-%m}",
                   "end": f"{table.index[-1]:%Y-%m}",
                   "months": int(len(table))},
        "data_notes": data_notes,
        "stationarity": stationarity,
        "single_variable_ranking": ranking,
        "selection_steps": steps,
        "selected_variables": selected,
        "model": None,
        "coefficients": [],
        "vif": [],
        "diagnostics": [],
        "flags": flags,
    }

    if not selected:
        flags.append("No variable improved Adjusted R2, so none of them helps "
                     "explain the target.")
        return result

    # Steps 5, 6, 7
    ols, summary, coefficients = fit_final_model(table, target, selected)
    vif = compute_vif(table, selected)
    diagnostics = run_diagnostics(ols)

    result.update(model=summary, coefficients=coefficients, vif=vif,
                  diagnostics=diagnostics)

    # Turn the technical results into plain-English flags.
    for c in coefficients:
        if c["variable"] != "const" and not c["significant_robust"]:
            flags.append(f"{c['variable']} improved Adjusted R2 but is not "
                         f"statistically significant (robust p="
                         f"{c['p_value_robust']:.3f}); treat it as weak evidence.")
    for v in vif:
        if v["vif"] > 10:
            flags.append(f"{v['variable']} has VIF {v['vif']:.1f} (>10): serious "
                         f"multicollinearity; consider dropping an overlapping variable.")
        elif v["vif"] > 5:
            flags.append(f"{v['variable']} has VIF {v['vif']:.1f} (>5): some "
                         f"overlap with other predictors.")
    for d in diagnostics:
        if not d["passed"]:
            flags.append(f"{d['test']} failed. {d['if_failed']}")
    if summary["f_test_p_value"] >= SIGNIFICANCE:
        flags.append("The overall F-test is not significant: the model as a "
                     "whole does not explain the target.")

    return result


# =============================================================================
# PRINT A READABLE REPORT (only used when running this file directly)
# =============================================================================
def print_report(r):
    print("\n=== DATA ===")
    print(f"Target: {r['target']} | Period: {r['period']['start']} to "
          f"{r['period']['end']} ({r['period']['months']} months)")
    for n in r["data_notes"]:
        print(" -", n)

    print("\n=== STATIONARITY ===")
    for s in r["stationarity"]:
        print(" -", s["message"])

    print("\n=== EACH VARIABLE ALONE (ranked by Adjusted R2) ===")
    for row in r["single_variable_ranking"]:
        print(f" {row['variable']:<12} R2={row['r2']:.4f}  AdjR2={row['adj_r2']:.4f}  "
              f"coef={row['coefficient']:.4f}  p={row['p_value']:.4f}")

    print("\n=== FORWARD SELECTION ===")
    for st in r["selection_steps"]:
        print(f" Step {st['step']}: added {st['added']:<12} AdjR2={st['adj_r2']:.4f} "
              f"(gain {st['gain_in_adj_r2']:+.4f})")
    print(" Selected:", r["selected_variables"] or "none")

    if r["model"]:
        m = r["model"]
        print("\n=== FINAL MODEL ===")
        print(f" n={m['n_observations']}  R2={m['r2']:.4f}  AdjR2={m['adj_r2']:.4f}  "
              f"F-test p={m['f_test_p_value']:.4g}  AIC={m['aic']:.1f}  BIC={m['bic']:.1f}")
        for c in r["coefficients"]:
            print(f" {c['variable']:<12} coef={c['coefficient']:+.5f}  "
                  f"p(classical)={c['p_value_classical']:.4f}  "
                  f"p(robust)={c['p_value_robust']:.4f}")

        print("\n=== MULTICOLLINEARITY (VIF) ===")
        for v in r["vif"]:
            print(f" {v['variable']:<12} VIF={v['vif']:.2f}")

        print("\n=== RESIDUAL DIAGNOSTICS ===")
        for d in r["diagnostics"]:
            p = f"p={d['p_value']:.4f}" if d["p_value"] is not None else ""
            stat = f"stat={d['statistic']:.3f}" if d["statistic"] is not None else ""
            print(f" {'PASS' if d['passed'] else 'FAIL'}  {d['test']}  {stat} {p}")

    print("\n=== FLAGS / WARNINGS ===")
    for f in r["flags"] or ["None"]:
        print(" -", f)


# =============================================================================
# SEARCH (used by the website's search box)
# =============================================================================
def search_series(query, limit=10):
    """Search FRED's catalogue by words. Returns a list of small dictionaries."""
    results = fred.search(query).head(limit)
    rows = []
    for code, row in results.iterrows():
        rows.append({
            "code": code,
            "title": row["title"],
            "frequency": row["frequency_short"],
            "units": row["units_short"],
            "start": str(row["observation_start"])[:10],
            "end": str(row["observation_end"])[:10],
        })
    return rows


if __name__ == "__main__":
    codes = ["CPIAUCSL", "UNRATE", "FEDFUNDS", "INDPRO"]
    print_report(run_analysis(codes, target="INDPRO"))