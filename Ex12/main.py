from typing import List

def compute_budget(total_budget: float,
                   citizen_votes: List[List[float]],
                   tol: float = 1e-100,
                   max_iter: int = 100) -> List[float]:
    """
    חישוב התקציב באמצעות אלגוריתם החציון המוכלל עם חיפוש בינארי על הסף τ.
    קלט:
      total_budget  – תקציב כולל float
      citizen_votes – רשימת רשימות, כל תושב מעריך כל סעיף
      tol           – דיוק לפסק זמן הבינארי
      max_iter      – מספר איטרציות מקסימלי
    פלט:
      רשימת התקציבים הסופיים לכל סעיף
    """
    n = len(citizen_votes) # מספר האזרחים - השורות במטריצה
    m = len(citizen_votes[0]) if n > 0 else 0 # מספר הסעיפים - מספר העמודות

    def median(lst: List[float]) -> float: # ביצוע חיפוש בינארי
        s = sorted(lst)
        k = len(s)
        mid = k // 2
        if k % 2 == 1:
            return s[mid]
        else:
            return 0.5 * (s[mid - 1] + s[mid])

    def b_j(j: int, tau: float) -> float: # לכל ערך בחציון מורידים tau ואז מוציאים חציון
        return median([min(citizen_votes[i][j], tau) for i in range(n)])

    # איתור הסף בחיפוש בינארי
    lo, hi = 0.0, max((max(row) for row in citizen_votes), default=0.0)
    for x in range(max_iter):
        mid = 0.5 * (lo + hi)
        total = sum(b_j(j, mid) for j in range(m))
        if total < total_budget:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    tau_star = 0.5 * (lo + hi)
    return [b_j(j, tau_star) for j in range(m)]


def compute_budget_direct(total_budget: float,
                          citizen_votes: List[List[float]]) -> List[float]:
    """
    חישוב התקציב בעזרת שיטה ישירה (ללא חיפוש בינארי),
    מבוססת על חלוקת g(τ) לפיסות קו עם שיפועים ידועים.
    קלט ופלט כמו ב–compute_budget.
    """
    n = len(citizen_votes)
    m = len(citizen_votes[0]) if n > 0 else 0

    # מיון הערכות לכל סעיף
    S = [sorted(citizen_votes[i][j] for i in range(n)) for j in range(m)]
    cuts = {0.0}
    odd = (n % 2 == 1)

    # איסוף כל נקודות השבר (Lj ו–Rj)
    # אם כמות באיברים אי זוגית - ניקח את האיבר האמצעי
    # אם כמות האיברים זוגית - לוקחים שני ערכים שביניהם חציון רציף
    for j in range(m):
        if odd:
            L = S[j][n // 2]
            cuts.add(L)
        else:
            L = S[j][n // 2 - 1]
            R = S[j][n // 2]
            cuts.add(L); cuts.add(R)

    pts = sorted(cuts)

    # מעבר על כל קטע [alpha, beta) חשב מקדמי g(τ)=a*τ + c
    # כי כל קטע בין שתי נקודות שכנות מייצג תחום שבו כל פונקצית החציון היא קטע קו לינארי
    for k in range(len(pts)):
        alpha = pts[k]
        beta = pts[k + 1] if k + 1 < len(pts) else float('inf')
        a = 0.0
        c = 0.0

        for j in range(m):
            if odd:
                L = S[j][n // 2]
                if alpha < L:
                    a += 1
                else:
                    c += L
            else:
                L = S[j][n // 2 - 1]
                R = S[j][n // 2]
                if alpha < L:
                    a += 1
                elif alpha < R:
                    a += 0.5
                    c += 0.5 * L
                else:
                    c += 0.5 * (L + R)

        if a > 0:
            tau = (total_budget - c) / a
            if alpha <= tau < beta:
                # מצא הקצאות סופיות לפי τ
                result = []
                for j in range(m):
                    if odd:
                        L = S[j][n // 2]
                        result.append(min(tau, L))
                    else:
                        L = S[j][n // 2 - 1]
                        R = S[j][n // 2]
                        if tau < L:
                            result.append(tau)
                        elif tau < R:
                            result.append(0.5 * (L + tau))
                        else:
                            result.append(0.5 * (L + R))
                return result

    # טיפול במצב שבו התקציב שווה בדיוק לסכום המדיאנים:
    #    (g(max_cut) == total_budget)
    # אם לא מצאנו פתרון בתוך אף קטע, ייתכן שהתקציב בדיוק שווה לסכום המדיאנים העליונים — אז פשוט מחזירים את אותם מדיאנים (הקצאה מקסימלית)
    max_alloc = []
    for j in range(m):
        if odd:
            max_alloc.append(S[j][n // 2])
        else:
            L = S[j][n // 2 - 1]
            R = S[j][n // 2]
            max_alloc.append(0.5 * (L + R))
    if abs(sum(max_alloc) - total_budget) < 1e-9:
        return max_alloc

    # אם לא נמצא פתרון — זרוק שגיאה
    raise ValueError("No feasible threshold found")


if __name__ == "__main__":
    # === שימוש ב-compute_budget (חיפוש בינארי) ===
    print("=== compute_budget (חיפוש בינארי) ===")
    b1 = compute_budget(100, [[100, 0, 0], [0, 0, 100]])
    print("compute_budget(100, [[100,0,0],[0,0,100]]) →", b1)
    # צפוי: [50.0, 0.0, 50.0]

    b2 = compute_budget(10, [[8, 2], [1, 9], [5, 5]])
    print("compute_budget(10, [[8,2],[1,9],[5,5]]) →", b2)
    # צפוי: [5.0, 5.0]

    # === שימוש ב-compute_budget_direct (שיטה ישירה) ===
    print("\n=== compute_budget_direct (שיטה ישירה) ===")
    d1 = compute_budget_direct(100, [[100, 0, 0], [0, 0, 100]])
    print("compute_budget_direct(100, [[100,0,0],[0,0,100]]) →", d1)
    # צפוי: [50.0, 0.0, 50.0]

    d2 = compute_budget_direct(10, [[8, 2], [1, 9], [5, 5]])
    print("compute_budget_direct(10, [[8,2],[1,9],[5,5]]) →", d2)
    # צפוי: [5.0, 5.0]
