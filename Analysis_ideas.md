# Sentiment Analysis Plan: Mastodon Reactions to GenAI Milestones (2022–2025)

## 1. Research Question & Sub-questions

**Main question**  
How did public sentiment toward generative AI on Mastodon change in response to major AI milestone events between 2022–2025?

**Sub-questions**

1. **Event impact**
   - Compared with the 7-day baseline before each event, how does sentiment change in the 7-day period after the event?
   - Are some events associated with strong positive shifts, others with negative shifts?

2. **Temporal dynamics**
   - Beyond the immediate 7-day window, do we see longer-term trends toward more optimism or skepticism about GenAI across events?
   - Are later events (e.g., 2024–2025) received differently than earlier ones, controlling for baseline sentiment?

3. **Instance / community heterogeneity**
   - How do different Mastodon instances react to the same event?
   - Are certain instances systematically more optimistic or more critical about GenAI?

4. **Content & topic differences**
   - For a given event, what themes dominate discussions (e.g., safety, regulation, jobs, creativity, technical issues)?
   - How are different themes associated with different sentiment patterns?

5. **User-level patterns (if data allow)**
   - Are reactions driven by a small core of highly active users, or by many casual users?
   - Do users consistently hold similar sentiment across events, or change their attitude over time?

---

## 2. Data Structure & Event Windows

2.2 Event windows and baseline definition

For each event with date `t0`:

- **Baseline period**: 7 days before the event  
  - `Baseline = [t0 − 7, t0 − 1]`
- **Post-event comparison period**: 7 days after the event  
  - `Post = [t0 + 1, t0 + 7]`
- (Event day `t0` can be analyzed separately or included in Post as a robustness check.)

All main comparisons use **Baseline vs Post** within the same event and instance.

---

## 3. Event-Level Analyses (Global Mastodon View)

**Goal:** Quantify how Mastodon as a whole reacts to each event.

For each `(event)` over Baseline and Post:

1. **Volume and participation**
   - Number of posts in Baseline vs Post.
   - (If available) number of unique users in Baseline vs Post.
   - Change in volume: `ΔN_posts = N_post − N_baseline`.

2. **Sentiment shift**
   - Mean sentiment polarity in Baseline vs Post.
   - Change in sentiment: `Δμ_polarity = μ_post − μ_baseline`.
   - Changes in sentiment class distribution:
     - Baseline vs Post shares of Positive / Neutral / Negative.

3. **Event typology**
   - Classify events by type (e.g., product launch, safety incident, regulation, scandal, open-source release).
   - Compare typical `Δμ_polarity` and `ΔN_posts` across event types:
     - Do safety or regulation events lead to stronger negative shifts?
     - Do product launches lead to stronger positive shifts?

4. **Temporal evolution across years**
   - For events ordered over time (2022 → 2025), analyze:
     - Trend in Baseline sentiment (is the “starting mood” toward GenAI changing?).
     - Trend in `Δμ_polarity` (are later events less exciting or more worrying?).

---

## 4. Instance- / Community-Level Analyses

**Goal:** Understand heterogeneity across Mastodon instances.

For each `(event, instance)`:

1. **Local impact**
   - Baseline vs Post:
     - `N_posts`, `N_users` (if possible).
     - `μ_polarity_baseline`, `μ_polarity_post`.
     - `Δμ_polarity_instance = μ_post − μ_baseline`.
     - Changes in sentiment class shares.

2. **Comparison to global Mastodon**
   - Compare each instance’s sentiment to the overall Mastodon average:
     - `Deviation_baseline = μ_polarity_instance_baseline − μ_polarity_global_baseline`.
     - `Deviation_post = μ_polarity_instance_post − μ_polarity_global_post`.

3. **Systematic tendencies**
   - Across events, identify instances that:
     - Are consistently more positive or more negative than the global average.
     - React more strongly to safety vs product events.
   - Aggregate metrics per instance:
     - Average `Δμ_polarity` across all events.
     - Variance of `Δμ_polarity` (how volatile the community is).

4. **Heterogeneity assessment**
   - Measure dispersion of instance-level sentiment (e.g., standard deviation of `μ_polarity_post` across instances for each event).
   - Ask whether heterogeneity grows or shrinks over time (later events more polarized across instances?).

---

## 5. Topic & Content Analyses

**Goal:** Explain *why* sentiment changes, not only *how much*.

### 5.1 Theme / topic identification

Use one or both of:

- **Keyword-based themes**  
  Build dictionaries for themes such as:
  - Safety & risk (e.g., “harm”, “danger”, “alignment”, “misuse”)
  - Regulation & policy (e.g., “EU AI Act”, “ban”, “law”)
  - Jobs & labor (e.g., “unemployment”, “automation”, “job loss”)
  - Creativity & art (e.g., “artist”, “style”, “plagiarism”, “copyright”)
  - Privacy & surveillance
  - Bias & fairness
  - Technical performance & reliability
  - Everyday usage / tips / prompts
  - Memes / humor / casual talk

- **Unsupervised topics**  
  Apply topic modeling to identify clusters, then manually label them using top words and sample posts.

Each post can receive one or multiple theme/topic labels.

### 5.2 Theme-conditional sentiment and volume

For each `(event, period ∈ {Baseline, Post}, theme)`:

1. **Theme prevalence**
   - Number and share of posts belonging to each theme.
   - Changes in theme composition from Baseline to Post:
     - Which themes become more prominent after the event?

2. **Theme-specific sentiment**
   - Mean sentiment polarity within each theme.
   - Changes in sentiment within themes from Baseline to Post:
     - Does a “jobs” theme turn more negative after certain events?
     - Do “usage/tips” posts stay positive even when other themes become negative?

3. **Event-type interaction**
   - For each event type (product, safety, regulation, etc.), analyze:
     - Typical theme mix (what people talk about).
     - Typical theme-level sentiment shifts (how they feel within those themes).

---

## 6. User-Level Analyses (If User IDs Are Available)

**Goal:** Distinguish between community-wide attitude shifts and behavior of a few highly active users.

For each `(event)`:

1. **Participation structure**
   - Distribution of posts per user in Baseline and Post:
     - Are reactions driven by many one-off contributors or a few heavy posters?

2. **Individual sentiment stability**
   - For users who post both in Baseline and Post:
     - Compare their mean sentiment before vs after the event.
     - Identify users whose sentiment shifts most strongly.

3. **Across-event behavior**
   - Track users across multiple events:
     - Are there “consistently positive” vs “consistently negative” users?
     - Are there users who become more skeptical over time?

---

## 7. Statistical Comparisons & Modeling

**Goal:** Provide quantitative evidence for observed patterns.

1. **Baseline vs Post comparisons (per event, overall Mastodon)**
   - Test differences in mean sentiment polarity.
   - Test differences in sentiment class distribution (Positive/Neutral/Negative).
   - Test differences in topic/theme shares.

2. **Instance-level effects**
   - For each event, test whether instances differ significantly in Post sentiment.
   - Model sentiment with instance fixed effects to capture systematic community tendencies.

3. **Event-type and time effects**
   - Build simple regression models where:
     - Outcome: post-level sentiment polarity.
     - Predictors:
       - `Event_type` (product, safety, regulation, etc.).
       - `Period` (Baseline vs Post).
       - `Year` or event order.
       - `Instance` (as fixed effects or random effects).
       - Interactions, e.g. `Event_type × Period`, `Event_type × Year`.
   - Use these models to answer:
     - Which types of events cause the largest sentiment shifts?
     - Are later events associated with systematically different reactions, controlling for event type and instance?

---

## 8. Final Storyline

1. **Describe** the overall Mastodon mood around GenAI before vs after key milestones.
2. **Quantify** how each event changes sentiment relative to its own 7-day baseline.
3. **Reveal** heterogeneity across instances and over time.
4. **Explain** sentiment shifts through content and topics (what people are actually talking about).
5. **Support** claims with basic statistical comparisons and simple models.