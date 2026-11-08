"""
Encodes:
1. Expected theoretical signs from Merton (1974) and Collin-Dufresne et al. (2001).
2. Macroeconomic primacy constraints: target credit spread cannot contemporaneously (tau=0)
   cause sovereign yields or market index returns, while leaving lagged links unconstrained.
"""

from typing import Dict, List, Optional


# Theoretical Expected Signs on spread changes (\Delta CS_t)
# Grounded in Merton (1974) and Collin-Dufresne, Goldstein, and Martin (2001)
EXPECTED_SIGNS: Dict[str, str] = {
    # --- Autoregressive & Target Spread ---
    "d_CS_BBB": "+",     # Autoregressive momentum / spread persistence
    "d_CS_i": "+",       # Individual bond spread persistence

    # --- Macroeconomic Shocks (Collin-Dufresne 2001) ---
    "d_DGS10": "-",      # Spot rate rise: higher risk-free drift lowers default prob -> narrows spread
    "d_T10Y2Y": "-",     # Curve steepening: expansionary signal -> narrows spread
    "ret_SPY": "-",      # Broad equity return: market-wide firm value increases -> narrows spread
    "d_VIX": "+",        # Implied asset volatility: raises default put option value -> widens spread
    "d_DEF_PREM": "+",   # Systemic default premium: tier widening / flight-to-quality -> widens spread

    # --- Micro / Firm-Level Drivers (Merton 1974) ---
    "daily_return": "-", # Firm-specific equity return: increases firm asset value V_A -> narrows spread
    "leverage": "+",     # Book leverage (Debt/Assets): decreases distance-to-default -> widens spread
    "daily_volume": "-", # Trading liquidity proxy: higher liquidity compresses liquidity premium
    "is_senior": "-",    # Seniority indicator: higher recovery expectation -> lower credit spread
}


def build_link_assumptions(
    var_names: List[str],
    target_var: str = "d_CS_BBB",
    tau_max: int = 5,
    strictly_exogenous: Optional[List[str]] = None,
) -> Dict:
    """
    Constructs the Tigramite `link_assumptions` dictionary enforcing macroeconomic primacy.

    Rules enforced:
    1. Contemporaneous Primacy (tau = 0):
       - The credit spread target cannot contemporaneously cause sovereign Treasury yields
         or broad equity indices.
       - Edges from Macro (i) -> Target (target_idx) at tau=0 are permitted.
       - Edges from Target (target_idx) -> Macro (i) at tau=0 are forbidden.
       - Edges between two non-target variables at tau=0 are unoriented.
    2. Lagged Feedback (1 <= tau <= tau_max):
       - All lagged relationships are unconstrained, allowing the data to reveal
         whether past credit shocks cause subsequent flight-to-safety or liquidity retrenchment.
    3. Strictly Exogenous Variables:
       - If specified, these variables cannot be caused by any other variable at any lag.

    Parameters:
    var_names : List[str]
        List of all variable names in the time series.
    target_var : str, default "d_CS_BBB"
        The dependent credit spread change variable.
    tau_max : int, default 5
        Maximum time lag to evaluate.
    strictly_exogenous : Optional[List[str]]
        Optional list of variable names that cannot be caused by any other variable.

    Returns:
    Dict
        Tigramite-compatible link assumptions dictionary.
    """
    if target_var not in var_names:
        raise ValueError(f"target_var '{target_var}' must be in var_names: {var_names}")

    n_vars = len(var_names)
    target_idx = var_names.index(target_var)
    exog_indices = set(var_names.index(v) for v in (strictly_exogenous or []) if v in var_names)

    link_assumptions = {j: {} for j in range(n_vars)}

    for j in range(n_vars):
        is_j_target = (j == target_idx)
        is_j_exog = (j in exog_indices)

        for i in range(n_vars):
            is_i_target = (i == target_idx)

            for tau in range(0, tau_max + 1):
                if i == j and tau == 0:
                    continue

                if tau == 0:
                    # Target cannot contemporaneously cause other variables
                    if is_j_target and not is_i_target:
                        link_assumptions[j][(i, 0)] = "-?>"
                    elif is_i_target and not is_j_target:
                        link_assumptions[j][(i, 0)] = "<?-"
                    else:
                        link_assumptions[j][(i, 0)] = "o?o"
                else:
                    # Lagged edges
                    if is_j_exog and i != j:
                        # Exogenous variables cannot have incoming lagged links from other vars.
                        # Tigramite convention: omitting the key means "no link".
                        continue
                    else:
                        link_assumptions[j][(i, -tau)] = "-?>"

    return link_assumptions

