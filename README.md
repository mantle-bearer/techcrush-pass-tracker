# PassPilot — Cohort Pass Tracker

> **A tiny, transparent score calculator to see if you’re on pace for the 80‑point pass mark.**

PassPilot helps students and tutors quickly estimate cumulative scores using the official cohort weights. It supports real‑world edge cases (missed vs not‑yet quizzes), validates inputs, and shows how many points you still need—plus the required average across remaining components.

---

## ✨ Features

* **Official weights baked in**: Quizzes **15**, Final **25**, Tutor **20**, Capstone **35**, Compliance **5** (out of 100).
* **Quiz status per row**: `Scored`, `Not yet` (excluded), `Missed (0)` (counts as zero but doesn’t “flatten” your whole grade).
* **Input validation**: Quizzes **0–20**, other categories **0–100**, with clear inline error messages.
* **Progress & projections**:

  * Points earned so far
  * Points needed to reach **80**
  * **Required average** across the remaining components you select
* **Privacy‑friendly**: Everything is calculated in your browser using **LocalStorage**. No accounts, no servers, no tracking.
* **Zero build**: Single `index.html` using Tailwind CDN + React UMD + Babel. Deploy anywhere as a static site (Render, Netlify, GitHub Pages, etc.).

---

## 📊 Scoring model (transparent math)

**Weights**

* Bi‑Weekly Quizzes (total) → **15 pts**
* Final Exam → **25 pts**
* Tutor Grading (assignments + participation + questions) → **20 pts**
* Capstone Project → **35 pts**
* Compliance (deadlines/instructions) → **5 pts**

**Formulas**

Let `qᵢ` be quiz scores (each **/20**), `n` = number of included quizzes.

* **Quiz % average**: $\text{quizAvg%} = \frac{1}{n}\sum\limits_{i=1}^{n} \min\big(\max(\frac{qᵢ}{20}, 0), 1\big) \times 100$
* **Quiz points**: $\text{quizPts} = (\text{quizAvg%} / 100) \times 15$
* **Final points**: $\text{finalPts} = \text{final%} / 100 \times 25$
* **Tutor points**: $\text{tutorPts} = \text{tutor%} / 100 \times 20$
* **Capstone points**: $\text{capstonePts} = \text{capstone%} / 100 \times 35$
* **Compliance points**: $\text{compliancePts} = \text{compliance%} / 100 \times 5$
* **Total**: $\text{total} = \text{quizPts} + \text{finalPts} + \text{tutorPts} + \text{capstonePts} + \text{compliancePts}$
* **Pass threshold**: $\text{total} \ge 80$
* **Required average across remaining**:
  Let `remainingCapacity` be the sum of max points for the selected remaining components.
  $\text{requiredAvg%} = \max\big(0, \min(100, \frac{80 - \text{total}}{\text{remainingCapacity}} \times 100)\big)$

**Edge cases**

| Case                | How PassPilot handles it                                                                       |
| ------------------- | ---------------------------------------------------------------------------------------------- |
| Missed quiz         | Mark row as **Missed (0)** → included as **0** in the average (fair impact, no global zeroing) |
| Not yet taken       | Mark row as **Not yet** → **excluded** from the average so it doesn’t drag you down            |
| Blank/invalid input | Shows inline error and ignores invalid value until fixed                                       |

---

## 🧪 Example

* Quizzes: 16/20, 18/20, Not yet, Missed → average is `(0.8 + 0.9 + 0 + (exclude)) / 3 = 56.7%`, so **8.50/15** points.
* Final: **72%** → **18.00/25** points.
* Tutor: **85%** → **17.00/20** points.
* Capstone: *(pending)* → **0/35** (for now).
* Compliance: **90%** → **4.50/5** points.
* **Total so far**: 8.50 + 18 + 17 + 0 + 4.5 = **48.00/100**.
* **Points needed to reach 80**: **32.00**. If only Capstone (35) remains, required average across remaining = `32/35 = 91.4%`.

---

## 🚀 Quick start

### Option A — Deploy on Render (Static Site)

1. Put the single file `index.html` at the **repo root**.
2. Push to GitHub.
3. On Render: **New → Static Site** → connect repo.

   * **Build Command:** *leave blank*
   * **Publish Directory:** `/`
4. Deploy. Done ✅

### Option B — Open locally

Just open `index.html` in any modern browser. No server needed.

### Option C — GitHub Pages

1. Repo settings → Pages.
2. Source: **Deploy from a branch**, choose branch + root folder.
3. Save → GitHub builds and hosts the page.

> The app uses Tailwind (CDN), React 18 (UMD), and Babel Standalone; there is **no build step**. If you prefer a bundler, see the **Vite** notes below.

---

## 🧰 Tech stack

* **React 18 UMD** + **ReactDOM 18 UMD**
* **Tailwind CSS** via CDN
* **Babel Standalone** for in‑browser JSX
* Data stored in **LocalStorage** only

---

## 🔐 Privacy

* No accounts, no backend, no analytics.
* All data is stored **locally** in your browser’s **LocalStorage**.
* Clear your browser storage or click **Reset all** inside the app to wipe data.

---

## ♿ Accessibility

* Numeric inputs have range hints and error messages.
* Status selector for each quiz supports keyboard and screen readers.
* Progress badges provide text alternatives.
* If you need more ARIA labels or full keyboard shortcuts, open an issue.

---

## 🛠 Customization

* **App name & logo**: Edit the `<title>`, header text, and the inline SVG favicon in `index.html`.
* **Colors**: Adjust Tailwind classes or inline styles in the same file.
* **Weights**: If your program changes weights, search for `15`, `25`, `20`, `35`, `5` and update the constants in the formulas and labels.

---

## 📦 Optional: Vite version (dev build)

If you prefer a typical React setup with a build step:

```bash
npm create vite@latest passpilot -- --template react
cd passpilot
npm install
```

Then move the JSX logic from this repo’s `index.html` into `src/App.jsx` and Tailwind into your Vite pipeline, or copy the earlier React component variant.

**Minimal `package.json` scripts**

```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  }
}
```

For **Render (Static)** with Vite, set:

* **Build Command:** `npm run build`
* **Publish Directory:** `dist`

---

## 🧩 Alternate component version

There’s a React component variant (original “80% Pass Tracker”) that assumes all quizzes are numeric and doesn’t distinguish `Not yet` vs `Missed`. If you need that minimal component, copy the file into your app and import it directly. The PassPilot UI is recommended for real cohorts because of better edge‑case handling.

---

## 🤝 Contributing

Issues and PRs are welcome! Ideas:

* CSV import/export of quiz rows
* Dark mode toggle
* Per‑component solver (e.g., “What do I need on Final alone to hit 80?”)
* Localization (copy deck and number formats)

---

## 📄 License

MIT — do anything, just keep the license.

---

## 🔗 Links

* **Live demo**: `https://your-deployed-url.example`
* **Issues**: open a ticket with steps to reproduce and screenshots if possible.

Happy piloting toward that 80+! 🛩️✅
