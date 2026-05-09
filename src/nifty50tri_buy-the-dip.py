import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# ============================================================
# KNOBS
# ============================================================
CSV_PATH        = Path(__file__).parent / "../data/nifty50tri_2000_2026.csv"
START_DATE      = "2000-01-01"
END_DATE        = "2026-05-08"
MONTHLY_AMOUNT  = 10_000
CASH_INTEREST_PA  = 0.07# Idle cash sits in a liquid debt fund
STEP_UP_PCT = 0.05      # annual SIP step-up (0 = off)


# --- Strategy 2: Buy-the-Dip ---
DIP_THRESHOLD_PCT = 3.5              # Catching only real crashes
REFERENCE_RESET   = "rolling_high"   # "after_buy" | "rolling_high" | "fixed"
DEPLOY_LEFTOVER   = True

# --- v3.3: tranching ---
TRANCHE_FRACTION    = 0.75   # fraction of pool deployed per dip buy
MAX_DIPS_PER_MONTH  = 2      # hard cap on dip buys per calendar month

# --- Strategy 3: Hybrid ---
HYBRID_SIP_AMOUNT = 7_500
HYBRID_DIP_AMOUNT = 2_500
# ============================================================


def load_data(path, start, end):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = (df[["Date", "TotalReturnsIndex"]]
            .rename(columns={"Date": "date", "TotalReturnsIndex": "close"})
            .dropna(subset=["close"])
            .sort_values("date"))
    df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    df["next_close"] = df["close"].shift(-1).fillna(df["close"])  # last day → same day
    return df


def first_trading_days(df):
    df = df.copy()
    df["ym"] = df["date"].dt.to_period("M")
    return df.groupby("ym", sort=True).first().reset_index(drop=True)


def xirr(cashflows):
    if len(cashflows) < 2:
        return None
    dates   = [cf[0] for cf in cashflows]
    amounts = [cf[1] for cf in cashflows]
    t0      = min(dates)
    years   = [(d - t0).days / 365.25 for d in dates]

    def npv(r):
        return sum(a / (1 + r) ** t for a, t in zip(amounts, years))

    try:
        return brentq(npv, -0.9999, 50.0, maxiter=1000)
    except Exception:
        return None

def step_factor(contrib_idx):
    """Contribution count → multiplier. 12 contribs = 1 year = ×(1+step)."""
    return (1 + STEP_UP_PCT) ** (contrib_idx // 12)

# ─────────────────────────────────────────────────────────────
# Strategy 1 – Regular SIP  (TRI: buys at T+1 close)
# ─────────────────────────────────────────────────────────────
def strategy_sip(df):
    monthly = first_trading_days(df)
    units, invested = 0.0, 0.0
    cashflows, buy_dates = [], []

    for i, (_, row) in enumerate(monthly.iterrows()):
        amount = MONTHLY_AMOUNT * step_factor(i)
        price  = row["next_close"]
        units    += amount / price
        invested += amount
        cashflows.append((row["date"], -amount))
        buy_dates.append(row["date"])

    final_price = df["close"].iloc[-1]
    final_value = units * final_price
    cashflows.append((df["date"].iloc[-1], final_value))

    return dict(units=units, invested=invested,
                final_value=final_value, cashflows=cashflows,
                buy_dates=buy_dates, buys=len(monthly), cash_idle=0.0)


# ─────────────────────────────────────────────────────────────
# Strategy 2 – Buy the Dip  (TRI: detect at close, buy at T+1 close)
# ─────────────────────────────────────────────────────────────
def strategy_buy_dip(df, threshold_pct=DIP_THRESHOLD_PCT,
                     ref_reset=REFERENCE_RESET, deploy_leftover=DEPLOY_LEFTOVER,
                     interest_pa=CASH_INTEREST_PA, monthly_amount=MONTHLY_AMOUNT,
                     tranche_fraction=TRANCHE_FRACTION,
                     max_dips_per_month=MAX_DIPS_PER_MONTH):
    contrib_dates = set(first_trading_days(df)["date"])

    cash_pool, units, invested = 0.0, 0.0, 0.0
    interest_earned = 0.0
    cashflows, buy_dates = [], []
    buy_count = 0
    ref_price = df["close"].iloc[0]
    prev_event_date = None
    current_month   = None
    dips_this_month = 0
    contrib_idx     = 0

    def accrue(today):
        nonlocal cash_pool, interest_earned, prev_event_date
        if prev_event_date is not None and cash_pool > 0 and interest_pa > 0:
            yrs = (today - prev_event_date).days / 365.25
            if yrs > 0:
                growth = (1 + interest_pa) ** yrs
                interest_earned += cash_pool * (growth - 1)
                cash_pool       *= growth
        prev_event_date = today

    for _, row in df.iterrows():
        date    = row["date"]
        close_p = row["close"]
        buy_p   = row["next_close"]

        # reset per-month counter on calendar rollover
        ym = date.to_period("M")
        if ym != current_month:
            current_month   = ym
            dips_this_month = 0

        # 1. contribution day → accrue then add cash
        if date in contrib_dates:
            accrue(date)
            amount = monthly_amount * step_factor(contrib_idx)
            cash_pool += amount
            invested  += amount
            cashflows.append((date, -amount))
            prev_event_date = date
            contrib_idx += 1

        # 2. update ref_price (using close)
        if ref_reset == "rolling_high":
            ref_price = max(ref_price, close_p)

        # 3. dip check — only if monthly cap not yet hit
        if dips_this_month < max_dips_per_month:
            drop_pct = (ref_price - close_p) / ref_price * 100.0
            if drop_pct >= threshold_pct and cash_pool > 0:
                accrue(date)
                deploy = cash_pool * tranche_fraction
                units     += deploy / buy_p
                cash_pool -= deploy
                buy_count += 1
                dips_this_month += 1
                buy_dates.append(date)
                prev_event_date = date
                if ref_reset in ("after_buy", "rolling_high"):
                    ref_price = close_p

    # final settlement
    final_date  = df["date"].iloc[-1]
    final_price = df["close"].iloc[-1]
    accrue(final_date)

    if cash_pool > 0 and deploy_leftover:
        units    += cash_pool / final_price
        buy_dates.append(final_date)
        cash_pool = 0.0

    final_value = units * final_price + cash_pool
    cashflows.append((final_date, final_value))

    return dict(units=units, invested=invested,
                final_value=final_value, cashflows=cashflows,
                buy_dates=buy_dates, buys=buy_count,
                cash_idle=cash_pool, interest_earned=interest_earned)


# ─────────────────────────────────────────────────────────────
# Strategy 3 – Hybrid  (TRI: detect at close, buy at T+1 close)
# ─────────────────────────────────────────────────────────────
def strategy_hybrid(df, sip_amount=HYBRID_SIP_AMOUNT, dip_amount=HYBRID_DIP_AMOUNT,
                    threshold_pct=DIP_THRESHOLD_PCT, ref_reset=REFERENCE_RESET,
                    deploy_leftover=DEPLOY_LEFTOVER, interest_pa=CASH_INTEREST_PA,
                    tranche_fraction=TRANCHE_FRACTION,
                    max_dips_per_month=MAX_DIPS_PER_MONTH):
    contrib_dates = set(first_trading_days(df)["date"])
    total_monthly = sip_amount + dip_amount

    cash_pool, units, invested = 0.0, 0.0, 0.0
    interest_earned = 0.0
    cashflows = []
    sip_buy_dates, dip_buy_dates = [], []
    dip_buy_count = 0
    ref_price = df["close"].iloc[0]
    prev_event_date = None
    current_month   = None
    dips_this_month = 0
    contrib_idx     = 0

    def accrue(today):
        nonlocal cash_pool, interest_earned, prev_event_date
        if prev_event_date is not None and cash_pool > 0 and interest_pa > 0:
            yrs = (today - prev_event_date).days / 365.25
            if yrs > 0:
                growth = (1 + interest_pa) ** yrs
                interest_earned += cash_pool * (growth - 1)
                cash_pool       *= growth
        prev_event_date = today

    for _, row in df.iterrows():
        date    = row["date"]
        close_p = row["close"]
        buy_p   = row["next_close"]

        # reset per-month counter on calendar rollover
        ym = date.to_period("M")
        if ym != current_month:
            current_month   = ym
            dips_this_month = 0

        # 1. contribution day → SIP slice buys, dip slice into pool
        if date in contrib_dates:
            accrue(date)
            sf      = step_factor(contrib_idx)
            sip_now = sip_amount * sf
            dip_now = dip_amount * sf
            if sip_now > 0:
                units += sip_now / buy_p
                sip_buy_dates.append(date)
            cash_pool += dip_now
            invested  += sip_now + dip_now
            cashflows.append((date, -(sip_now + dip_now)))
            prev_event_date = date
            contrib_idx += 1

        # 2. ref update
        if ref_reset == "rolling_high":
            ref_price = max(ref_price, close_p)

        # 3. dip check (daily) — only if monthly cap not yet hit
        if dips_this_month < max_dips_per_month:
            drop_pct = (ref_price - close_p) / ref_price * 100.0
            if drop_pct >= threshold_pct and cash_pool > 0:
                accrue(date)
                deploy = cash_pool * tranche_fraction
                units         += deploy / buy_p
                cash_pool     -= deploy
                dip_buy_count += 1
                dips_this_month += 1
                dip_buy_dates.append(date)
                prev_event_date = date
                if ref_reset in ("after_buy", "rolling_high"):
                    ref_price = close_p

    final_date  = df["date"].iloc[-1]
    final_price = df["close"].iloc[-1]
    accrue(final_date)

    if cash_pool > 0 and deploy_leftover:
        units    += cash_pool / final_price
        dip_buy_dates.append(final_date)
        cash_pool = 0.0

    final_value = units * final_price + cash_pool
    cashflows.append((final_date, final_value))

    return dict(units=units, invested=invested,
                final_value=final_value, cashflows=cashflows,
                buy_dates=sip_buy_dates + dip_buy_dates,
                sip_buy_dates=sip_buy_dates, dip_buy_dates=dip_buy_dates,
                buys=len(sip_buy_dates) + dip_buy_count,
                sip_buys=len(sip_buy_dates), dip_buys=dip_buy_count,
                cash_idle=cash_pool, interest_earned=interest_earned)


# ─────────────────────────────────────────────────────────────
# Plotting – 3 subplots
# ─────────────────────────────────────────────────────────────
def plot_buys(df, sip, dip, hyb):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    panels = [
        (axes[0], [(sip["buy_dates"], "tab:blue", 0.25)],
         f"Regular SIP — {sip['buys']} buys"),
        (axes[1], [(dip["buy_dates"], "tab:red", 0.25)],
         f"Buy-the-Dip — {dip['buys']} buys"),
        (axes[2], [(hyb["sip_buy_dates"], "tab:blue", 0.18),
                   (hyb["dip_buy_dates"], "tab:red",  0.45)],
         f"Hybrid — {hyb['sip_buys']} SIP + {hyb['dip_buys']} dip buys"),
    ]

    for ax, layers, title in panels:
        ax.plot(df["date"], df["close"], color="black", lw=0.8)
        for dates, color, alpha in layers:
            for d in dates:
                ax.axvline(d, color=color, alpha=alpha, lw=0.6)
        ax.set_title(title)
        ax.set_ylabel("Nifty 50 TRI")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
# Indian number formatting
# ─────────────────────────────────────────────────────────────
def ind(n, decimals=0):
    negative = n < 0
    n = abs(n)
    if decimals:
        integer_part = int(n)
        frac = f"{n - integer_part:.{decimals}f}"[1:]
    else:
        integer_part = int(round(n))
        frac = ""

    s = str(integer_part)
    if len(s) <= 3:
        formatted = s
    else:
        last3 = s[-3:]
        rest  = s[:-3]
        groups = []
        while rest:
            groups.append(rest[-2:] if len(rest) >= 2 else rest)
            rest = rest[:-2]
        formatted = ",".join(reversed(groups)) + "," + last3

    return ("-" if negative else "") + "₹" + formatted + frac


# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────
W = 58

def row(label, value=""):
    content = f"  {label:<22}{value}"
    return f"│{content:<{W}}│"

def divider(char="─"):
    return f"├{'─' * W}┤"

def report(label, res, df):
    start  = df["date"].iloc[0]
    end    = df["date"].iloc[-1]
    years  = (end - start).days / 365.25

    iv     = res["invested"]
    fv     = res["final_value"]
    gain   = fv - iv
    pct    = (fv / iv - 1) * 100 if iv else 0
    cagr   = ((fv / iv) ** (1 / years) - 1) * 100 if iv else 0
    xi     = xirr(res["cashflows"])
    xi_pct = xi * 100 if xi else float("nan")

    top    = f"┌{'─' * W}┐"
    bot    = f"└{'─' * W}┘"
    title  = f"│  {label:<{W - 2}}│"

    print(f"\n{top}")
    print(title)
    print(divider())
    print(row("Period", f"{start.date()} → {end.date()} ({years:.1f} yr)"))
    if "sip_buys" in res:
        print(row("Buys executed",
                  f"{res['buys']}  ({res['sip_buys']} SIP + {res['dip_buys']} dip)"))
    else:
        print(row("Buys executed", str(res["buys"])))
    print(divider())
    print(row("Total invested", ind(iv)))
    print(row("Final value",    ind(fv)))
    print(row("Absolute gain",  f"{ind(gain)}  ({pct:.1f}%)"))
    print(divider())
    print(row("CAGR",  f"{cagr:.2f}%"))
    print(row("XIRR",  f"{xi_pct:.2f}%"))
    if res.get("interest_earned", 0) > 0:
        print(row("Interest on idle cash", ind(res["interest_earned"])))
    if res["cash_idle"] > 0:
        print(divider())
        print(row("Cash never deployed", ind(res["cash_idle"])))
    print(bot)


# ─────────────────────────────────────────────────────────────
def main():
    df = load_data(CSV_PATH, START_DATE, END_DATE)
    print(f"\nLoaded {len(df)} trading days  |  "
          f"{df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
    print(f"Start price: {ind(df['close'].iloc[0], 2)}   "
          f"End price: {ind(df['close'].iloc[-1], 2)}")
    print(f"Tranche: {TRANCHE_FRACTION:.0%} of pool per dip  |  "
          f"Max {MAX_DIPS_PER_MONTH} dip buys/month  |  "
          f"Step-up: {STEP_UP_PCT:.0%}/yr")

    sip = strategy_sip(df)
    dip = strategy_buy_dip(df)
    hyb = strategy_hybrid(df)

    report("Regular SIP  (₹10k every month, no timing)", sip, df)
    report(f"Buy-the-Dip  (>{DIP_THRESHOLD_PCT}% drop, "
           f"{TRANCHE_FRACTION:.0%}/dip, ≤{MAX_DIPS_PER_MONTH}/mo)", dip, df)
    report(f"Hybrid  (₹{HYBRID_SIP_AMOUNT:,} SIP + ₹{HYBRID_DIP_AMOUNT:,} dip-pool)",
           hyb, df)

    sip_x = (xirr(sip["cashflows"]) or 0) * 100
    dip_x = (xirr(dip["cashflows"]) or 0) * 100
    hyb_x = (xirr(hyb["cashflows"]) or 0) * 100

    print("\n  Comparison vs SIP:")
    print(f"    Dip    → {ind(dip['final_value'] - sip['final_value']):>14}  "
          f"final  |  XIRR Δ {dip_x - sip_x:+.2f}%")
    print(f"    Hybrid → {ind(hyb['final_value'] - sip['final_value']):>14}  "
          f"final  |  XIRR Δ {hyb_x - sip_x:+.2f}%\n")

    plot_buys(df, sip, dip, hyb)


if __name__ == "__main__":
    main()
