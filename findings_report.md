# What actually runs on weak hardware?
## How much does an AI model's size decide how capable it is — and what else does?

**Problem.** Choose one open AI language model for a device with 4, 8 or 16 GB of memory. Test the rule 'pick the biggest model that fits' and replace it if it fails.

**Data.** Hugging Face dataset 'open-llm-leaderboard/contents'. 4,576 rows → 4,486 models after removing 10 with no valid size or score, 1 flagged and 79 duplicates. 2,823 are still downloadable.

**Assumptions.** 4-bit memory = 0.55 GB per billion parameters; a model may use 80% of device memory → 4 GB fits ≤5.8B, 8 GB ≤11.6B, 16 GB ≤23.3B.

## Conclusions

**1. Size matters, but on its own it is a partial rule.** Every 10× increase in size adds about 13.8 points, yet size explains only 40% of the differences between models (Figures 7–10).

**2. Which model you pick at a given size matters as much as a big jump in size.** A typical model lands 8.4 points away from what its size predicts — the same as making it about 4.0× bigger (Figure 11).

**3. How a model was trained, its blueprint and its date explain only a little more.** Adding them moves the explained share from 66% to 75%; whatever is still unexplained comes from unrecorded details such as the exact training data and recipe. The most important factors: Size, When it was made, How it was made (type) (Figures 12–17).

**4. Skills differ in how much they need size.** Hard reasoning puzzles (BBH) depends most on size; for Instruction following (IFEval) size is a poor guide, so choose by that test's own score (Figures 18–22).

**5. Small models are catching up.** Today's best 4 GB model matches what models of 30B or more scored about 17 months earlier (Figures 23–24).

**6. The better rule: within your memory budget, pick the best-scoring model that fits — not the biggest.** Best picks: 4 GB → **3Bgeneralv2-ECE-PRYMMAL-Martial** (31.5, vs 21.5 for a typical biggest-that-fits model); 8 GB → **Chocolatine-Fusion-14B** (40.4, vs 23.2 for a typical biggest-that-fits model); 16 GB → **T3Q-qwen2.5-14b-v1.0-e3** (47.1, vs 38.2 for a typical biggest-that-fits model) (Figures 25–28).

**7. Size is expensive.** Each 10× in size costs about 3× more energy for 13.8 more points — another reason to choose the smallest model that does the job (Figure 29).

**Limits of this analysis.** (a) The leaderboard is a frozen snapshot (last test: March 2025); newer models are not in it. (b) Scores are from six academic tests, not your real task — always try the shortlist on your own job. (c) Some community models, especially merges, may have been tuned to these exact tests, which inflates their scores; official releases are the safer choice. (d) Memory is estimated at 0.55 GB per billion parameters at 4-bit; real use also needs room for longer conversations. (e) Traits like blueprint and date are tangled together, so Part 3 shows associations, not proof of cause.

---

## Part 1 · Meet the data

Before answering anything, look at what we have: how big the models are, how well they score, how they were made, what they are built on, and when they appeared. These six charts set the scale for everything after them.

### Figure 1 · How big are the models?

![Figure 1](figures/fig01_size_distribution.png)

**What this graph shows.** How many models of each size the leaderboard contains. Size is in billions of parameters — the adjustable numbers that store what a model learned. Bars are coloured by size group; the black lines mark the largest model that fits on a 4, 8 and 16 GB device once compressed to 4-bit.

**How to read it.** The horizontal axis is stretched so each step to the right multiplies the size instead of adding to it: going from 1B to 10B takes the same room as 10B to 100B. A taller bar means more models of that size.

**What the data says.** The most crowded group is **Medium 3.5–9.5B** with **47%** of all models. The three most common exact sizes are **8B** (967 models), **7.6B** (382 models), **14.8B** (310 models). Only **22%** of models are 3.5B or smaller, and **25%** would fit a 4 GB device.

**Why it matters.** People build for the hardware they own — mostly one consumer graphics card — which is why the middle is so crowded. Where bars are short (the tiniest and the biggest models) our conclusions rest on fewer examples and deserve a little more caution.

### Figure 2 · How well do the models score?

![Figure 2](figures/fig02_score_distribution.png)

**What this graph shows.** How the overall leaderboard scores are spread out. Each score is the average of six hard tests, rescaled so that 0 means 'no better than random guessing' and 100 means 'perfect'.

**How to read it.** Each bar counts the models that scored inside that range. The black curve traces the overall shape. The shaded strip holds the middle half of all models; the coloured lines mark the average and the middle model.

**What the data says.** The average model scores **21.9** and the middle model **22.0**. Half of all models fall between **13.9** and **29.3**. The best reaches **52.1**, while **18%** of models score under 10 — barely better than guessing on these tests.

**Why it matters.** Because the tests are deliberately hard, even the best open models land around 50, not 90. A gap of a few points between two models is therefore a real difference — this sets the scale for every comparison below.

### Figure 3 · How were the models made?

![Figure 3](figures/fig03_model_types.png)

**What this graph shows.** How each model was made. **Pretrained (base)** models are trained from scratch by a lab. The rest start from one of those: **fine-tuned** (extra training on specific data), **chat-tuned** (trained to hold a conversation and follow instructions), and **merges** (the insides of several models blended into one).

**How to read it.** Longer bar = more models of that kind. The label gives the count and its share of the whole leaderboard.

**What the data says.** **Fine-tuned** is the biggest group (1,746, 39%), followed by **Merge** (38%). Only **6%** are original base models — the leaderboard is mostly community variations on a small number of lab releases.

**Why it matters.** When we compare kinds of models later, remember that many of them are close relatives of the same few base models. That is useful (lots of evidence about what extra training does) and a caution (one base model can appear hundreds of times).

### Figure 4 · Which blueprints are the models built on?

![Figure 4](figures/fig04_families.png)

**What this graph shows.** Which blueprint (architecture) each model is built on. The blueprint is the model's internal design. Organisations often reuse one another's blueprints — many models that are not from Meta still use the Llama design.

**How to read it.** Each dot's distance from the left is the number of models built on that blueprint. Each blueprint keeps the same colour in every later chart.

**What the data says.** **Llama** (41%) and **Qwen** (31%) together account for **72%** of the leaderboard. Everything outside the top 7 adds up to 4%.

**Why it matters.** Blueprint is one of the factors tested in Part 3: if two models are the same size, does the design they are built on change how good they are?

### Figure 5 · When did the models appear?

![Figure 5](figures/fig05_timeline.png)

**What this graph shows.** How many models were uploaded each month, split by how they were made.

**How to read it.** Each coloured layer is one kind of model; the top edge of the whole stack is that month's total. The date is when a model was first uploaded; where that is missing, the date it was tested is used instead. The final, partly-covered month is left out so it does not look like a collapse.

**What the data says.** The busiest month was **January 2025** with **784** models. Merges made up **27%** of uploads before Dec 2024 and **50%** after it.

**Why it matters.** The field moves fast, and much of the later activity is remixing rather than new base models. Part 5 measures whether that activity actually raised the best score available at each size.

### Figure 6 · What other traits do the models have?

![Figure 6](figures/fig06_traits.png)

**What this graph shows.** The share of models that have each of six yes/no traits.

**How to read it.** The blue part of each bar is the share that answers 'yes'. A full bar would mean every model has the trait.

**What the data says.** **37%** of the tested models have since been removed from Hugging Face. Only **10%** come from the original labs, **53%** are ready for chat, and **1.6%** use a mixture-of-experts design (only part of the model works on each word, although the whole model still has to fit in memory). Note that the 'labelled as a merge' flag is filled in less often than the 'Merge' type in Figure 3 — the two fields were recorded separately.

**Why it matters.** Because so many models have disappeared, every recommendation in Part 6 uses only models you can still download. Each trait here is also tested in Part 3: does it make a model better than its size suggests?

## Part 2 · Does size decide capability?

This is the central question. If you only know a model's size, how well can you guess its score? If the answer is *very well*, the rule 'pick the biggest model that fits' is good enough. If not, we need something better. The tool here is a straight line through the data: **score = start + slope × (how many times ×10 the size is)**. The slope tells us how many points each ten-fold increase in size buys, and **R²** (a number from 0% to 100%) tells us how much of the difference between models that line explains.

### Figure 7 · THE MAIN CHART: size against score

![Figure 7](figures/fig07_size_vs_score_main.png)

**What this graph shows.** Every model as one dot: how big it is (left to right) against how well it scores (bottom to top). The black line is the best straight-line guess of score from size alone. The purple line shows the actual typical score at each size, so you can see where the straight line fits and where it does not.

**How to read it.** Dots high up are good; dots far right are big. A dot **above** the black line does better than its size predicts; **below** does worse. The shaded zones show which device each size fits on.

**What the data says.** Each time a model gets 10 times bigger, its score rises by about **13.8 points**. But size alone explains only **40%** of the differences between models; the other **60%** comes from something else. Look at the 7–8B models alone: their scores run from **7** to **34** (middle 90% of them). The rank agreement between size and score is 0.63 on a scale where 1 would mean 'bigger is always better'.

**Why it matters.** If size explained nearly everything, 'pick the biggest that fits' would be the right rule. The wide vertical spread at every size says the opposite: **which** model you choose at a given size matters a lot. Parts 3 and 6 find out what drives that spread and turn it into a better rule.

### Figure 8 · Where are most of the models?

![Figure 8](figures/fig08_density_hexbin.png)

**What this graph shows.** The same data as the main chart, but instead of drawing thousands of overlapping dots it counts how many models fall inside each small hexagon. This reveals where models pile up, which overlapping dots hide.

**How to read it.** Darker blue means more models in that spot; pale means only a few. The orange line is the size-only prediction.

**What the data says.** The single most crowded region is models of **6.3B–12B** scoring **22–26**, holding **530** models. Even inside that one narrow size range, the dark area stretches up and down across many points.

**Why it matters.** The crowd at one size is the place where 'size' can't tell models apart at all, so other factors must decide which is better. It also shows the main chart's average line is set mostly by mid-size models.

### Figure 9 · Scores within each size group

![Figure 9](figures/fig09_violin_size_groups.png)

**What this graph shows.** For each size group, the full shape of its scores. A violin is wide where many models score and thin where few do. The three lines inside each violin split its models into four equal-sized quarters; the middle line is the median.

**How to read it.** Compare heights: higher violins = better groups. Then compare how much neighbouring violins overlap vertically — overlap means a model from the smaller group can beat one from the bigger group.

**What the data says.** The middle score of each group is **Tiny 7.3**, **Small 14.2**, **Medium 22.5**, **Large 32.0**, **X-Large 27.3**, **Huge 36.0**. Inside the Medium group alone, scores run from **5** to **33** (middle 90%).

**Why it matters.** Bigger groups do score higher on average, so size is not irrelevant. But the heavy overlap means a well-chosen model from a smaller group often matches a typical model from a bigger one — the next chart puts a number on it.

### Figure 10 · How often does a smaller model beat a bigger one?

![Figure 10](figures/fig10_smaller_beats_bigger.png)

**What this graph shows.** For each pair of neighbouring size groups, the share of models in the smaller group that score higher than the middle (median) model of the bigger group.

**How to read it.** Each bar is one comparison. 0% would mean the smaller group never wins; 50% would mean the two groups are equally good. The number in white at the bottom is the score the smaller models had to beat.

**What the data says.** **1%** of Tiny models beat the typical Small model; **13%** of Small models beat the typical Medium model; **7%** of Medium models beat the typical Large model; **60%** of Large models beat the typical X-Large model; **25%** of X-Large models beat the typical Huge model.

**Why it matters.** This is the first direct evidence against 'biggest that fits'. If a meaningful share of smaller models beat the typical bigger one, then choosing well within a smaller size can match or beat choosing blindly in a bigger one — which also means you could run a better model on cheaper hardware.

## Part 3 · Beyond size: what explains the rest?

For every model we now compute its **beat-the-curve score**: its actual score minus the score its size predicts (the black line in Figure 7). Positive = better than its size suggests, negative = worse. Then we ask which traits push models above the line. Finally a machine-learning model (a *random forest*: hundreds of simple decision trees that vote together) predicts scores from all traits at once, so we can measure how much each trait adds beyond size.

### Figure 11 · How far from the size prediction do models land?

![Figure 11](figures/fig11_beat_the_curve_histogram.png)

**What this graph shows.** For every model, the gap between its real score and the score the size-only line predicted for it.

**How to read it.** The black centre line means 'exactly what its size predicts'. Blue bars to the right are models that do better than their size; red bars to the left do worse. The grey lines mark the typical distance from the centre.

**What the data says.** A typical model lands **8.4 points** away from its size prediction. **13%** of models beat it by more than 10 points and **12%** fall short by more than 10. For comparison, making a model 10× bigger buys only **13.8 points** — so landing one typical step above the line is worth about as much as a model **4.0 times bigger**.

**Why it matters.** This is the key number of the whole project. The choice *between* models of the same size is worth as much as a large jump in size, so the selection rule must look past size.

### Figure 12 · Does HOW a model was made matter?

![Figure 12](figures/fig12_beat_curve_by_type.png)

**What this graph shows.** The beat-the-curve score for each kind of model. Each box covers the middle half of that kind's models; the line inside the box is the middle model. Faint grey dots are individual models.

**How to read it.** Boxes to the right of the black line beat their size; boxes to the left fall short. Rows are sorted from best (top) to worst (bottom).

**What the data says.** The middle model of each kind lands at: **Merge +1.6**, **Fine-tuned -0.1**, **Chat-tuned -0.3**, **Other -5.7**, **Pretrained (base) -6.4**. Base models are a useful reference: they are what the labs released before anyone tuned them.

**Why it matters.** This shows how much extra training is worth at a fixed size. If tuned and merged models sit clearly above base models, the training a model got is a bigger lever than adding parameters. Merges deserve one caution: some are tuned specifically to score well on this leaderboard.

### Figure 13 · Does the blueprint matter?

![Figure 13](figures/fig13_beat_curve_by_family.png)

**What this graph shows.** For each blueprint, the average beat-the-curve score of the models built on it.

**How to read it.** Bars to the right beat their size on average; bars to the left fall short. The whiskers show the range the true average probably lies in (95% uncertainty): if a whisker crosses the black line, that blueprint may really be no different from average.

**What the data says.** **Qwen** models average **+3.5** points against their size, while **GPT-NeoX** models average **-13.0**. The whiskers of the top and bottom blueprints do not overlap, so the gap is real, not luck.

**Why it matters.** Blueprint is tangled up with time: newer blueprints come from newer, better-trained releases. So a high bar here means 'models built on this design were better than their size' — partly design, partly the newer training data that came with it. Part 3's forest model separates these by looking at both at once.

### Figure 14 · Which yes/no traits push a model above its size?

![Figure 14](figures/fig14_trait_effects.png)

**What this graph shows.** For six yes/no traits, how much better (or worse) models WITH the trait do against their size compared with models WITHOUT it.

**How to read it.** Blue bars to the right: having the trait goes with beating your size. Red bars to the left: it goes with falling short. Whiskers crossing the black line mean the effect could be luck.

**What the data says.** **Ready for chat (chat template)**: +2.8 points; **Re-trained 2+ times on top of a base**: +0.8 points; **Still downloadable today**: -1.4 points; **Released by the original lab**: -4.9 points; **Published as an add-on (adapter)**: -7.0 points; **Mixture-of-experts design**: -9.4 points.

**Why it matters.** These are associations, not proof of cause. For example, deleted models may score worse simply because authors removed their failed experiments. Mixture-of-experts models are counted by their full size even though only part works at a time, which makes them look weak for their size — but they still need memory for all of it, so for a memory budget that penalty is real.

### Figure 15 · How much do all the traits together explain?

![Figure 15](figures/fig15_model_comparison.png)

**What this graph shows.** Three ways of predicting a model's score, each tested on a quarter of the models that were kept aside and never used to build the predictor (so the test is fair).

**How to read it.** Taller bar = better predictions. The first bar is our straight line. The second is a flexible forest model that still only sees size — it shows whether a curve beats a straight line. The third sees everything.

**What the data says.** Size alone explains **38%** with a straight line and **66%** with the flexible model, so the straight line was not the problem. Adding every other trait lifts it to **75%** — a gain of **9 percentage points**.

**Why it matters.** If the third bar is much taller than the second, then knowing how a model was made, its blueprint and its date really does help pick a better model. Whatever remains unexplained comes from things the leaderboard doesn't record, such as the exact training data and recipe.

### Figure 16 · Which traits matter most?

![Figure 16](figures/fig16_feature_importance.png)

**What this graph shows.** How important each trait is to the forest's predictions. We scramble one trait at a time (shuffle it randomly between models so it carries no information) and see how much worse the predictions get. Each trait's damage is then shown as its share of the total damage, so the bars add up to 100%.

**How to read it.** Longer bar = the forest leans on that trait more; '≈0%' means the forest ignores it. Colours group the traits: orange = size, blue = how it was trained, aqua = design, violet = time and format. Whiskers show how much the result varied over five scrambles.

**What the data says.** Share of the forest's reliance: **Size** 72%, **When it was made** 13%, **How it was made (type)** 4%, **Blueprint (family)** 4%, **Ready for chat** 3%, **Times re-trained** 2%.

**Why it matters.** This ranks the levers for choosing a model. If size is first but training traits follow closely, the selection rule should weigh both. Traits near zero can be ignored when choosing.

### Figure 17 · How accurate is the forest model?

![Figure 17](figures/fig17_predicted_vs_actual.png)

**What this graph shows.** For each held-back model, the score the forest predicted (left to right) against the score it actually got (bottom to top).

**How to read it.** A perfect predictor would put every dot on the black diagonal. Dots inside the blue strip were predicted within the average error. Dots far above the line did much better than their traits suggested.

**What the data says.** On average the forest's guess is off by **3.5 points**, and for half the models by less than **2.2**. Compare with the 8.4-point typical error of the size-only line.

**Why it matters.** A predictor this accurate could help screen new models before testing them. The dots far from the line are the surprises — models whose quality came from something the leaderboard doesn't record.

## Part 4 · Inside the average: which skills come from size?

The leaderboard score averages six very different tests. Here we take the average apart: do all six skills grow with size equally, or do some come from size and others from training? **Instruction following (IFEval)** — can it obey exact instructions such as 'answer in exactly three bullet points'. **Hard reasoning puzzles (BBH)** — can it solve tricky multi-step logic and reasoning puzzles. **Competition maths (MATH lvl 5)** — can it solve the hardest high-school competition maths problems. **PhD-level science (GPQA)** — can it answer graduate-level biology, physics and chemistry questions. **Long-story reasoning (MuSR)** — can it work out the answer to murder-mystery style puzzles hidden in long stories. **Broad knowledge (MMLU-PRO)** — does it know facts and concepts across 14 school and university subjects.

### Figure 18 · Which measures move together?

![Figure 18](figures/fig18_correlation_heatmap.png)

**What this graph shows.** How strongly each pair of measures rises and falls together across all models (a rank correlation).

**How to read it.** Each square compares the row with the column. **+1** (deep blue) means that when one is higher the other is always higher too; **0** (grey) means no link; red would mean opposite directions. Read the 'Size' row to see how strongly each skill is tied to size.

**What the data says.** The two skills most tied together are **Reasoning** and **Knowledge** (0.95); the least tied are **Instructions** and **Story logic** (0.37). Against size, **Reasoning** has the strongest link (0.70) and **Instructions** the weakest (0.39).

**Why it matters.** Skills that move together probably come from the same source. A skill weakly linked to size must come from somewhere else — usually from how the model was trained.

### Figure 19 · How each skill grows with size

![Figure 19](figures/fig19_skills_vs_size.png)

**What this graph shows.** Six small charts, one per test, each showing that test's score against model size.

**How to read it.** Faint dots are models; the coloured line traces the typical score at each size; the black line is the straight-line fit. The box in each corner gives the points gained per 10× size and how much of that test's differences size explains.

**What the data says.** Points gained per 10× size: **Instructions +17.4**, **Reasoning +21.2**, **Maths +12.8**, **Science +5.9**, **Story logic +6.5**, **Knowledge +19.2**.

**Why it matters.** Different skills respond to size very differently, so 'bigger is better' is true for some jobs and nearly irrelevant for others. The next chart ranks them.

### Figure 20 · Which skills depend on size?

![Figure 20](figures/fig20_skills_ranked_by_size.png)

**What this graph shows.** The six tests ranked by how much size alone explains each one.

**How to read it.** Longer bar = that skill depends more on size. The label also shows the points gained per 10× size.

**What the data says.** **Hard reasoning puzzles (BBH)** depends most on size (size explains **47%**), and **Instruction following (IFEval)** least (**18%**).

**Why it matters.** This splits the decision by use-case. If your job mostly needs **reasoning** (can it solve tricky multi-step logic and reasoning puzzles), size is hard to replace: a bigger budget pays off. If it mostly needs **instructions** (can it obey exact instructions such as 'answer in exactly three bullet points'), size is a poor guide — choose by that test's own score instead. One caution: a low share can also mean the test itself is noisy (scores close to random guessing jump around), so check Figure 18 to see whether that test agrees with the others.

### Figure 21 · The skill profile of each size group

![Figure 21](figures/fig21_radar_profiles.png)

**What this graph shows.** A spider chart: each spoke is one test, and each coloured shape is one size group's average on all six tests.

**How to read it.** On each spoke, 100% (the outer ring) is the best size group's average on that test; every other group is drawn as a share of it. A shape that is round means balanced skills; a dented shape means that group is weak on the dented spoke.

**What the data says.** The smallest group (**Tiny ≤1B**) reaches **36%** of the top level on **Instructions** but only **12%** on **Maths**.

**Why it matters.** Small models are not uniformly 'worse' — they are worse at some things and nearly as good at others. Matching the model to the job lets a small device do more than its size suggests.

### Figure 22 · Why the overall score can mislead

![Figure 22](figures/fig22_same_score_different_skills.png)

**What this graph shows.** Two models (both still downloadable) whose overall scores are almost identical, compared test by test. They were found by searching all pairs within half a point of each other for the most different skill profiles.

**How to read it.** Each pair of bars is one test. If the overall score told the whole story, the violet and aqua bars would be about the same height everywhere.

**What the data says.** **EXAONE-3.5-2.4B-Instruct** scores **27.1** overall and **PHI-4-Hindi** **27.5** — yet on **Instruction following (IFEval)** they differ by **79 points**.

**Why it matters.** Choosing by the overall score alone can hand you a model that is weak at exactly the skill you need. Pick on the overall score first, then check the one or two tests closest to your actual job.

## Part 5 · Are small models catching up?

A model's quality depends not just on its size but on when it was made — training methods and data keep improving. Here we track the best score available in each size group over time.

### Figure 23 · How the best model in each size group improved

![Figure 23](figures/fig23_record_by_size_over_time.png)

**What this graph shows.** For each size group, the best score any model had reached by each month.

**How to read it.** A line only ever goes up or stays flat. Each step up is a new record for that size group. Numbers at the right end are each group's best score at the end of the data.

**What the data says.** From the start of 2024 to the end of the data: **Tiny** went from 6.6 to 22.3; **Small** went from 15.5 to 31.5; **Medium** went from 22.8 to 40.4; **Large** went from 23.6 to 47.1; **X-Large** went from 27.2 to 47.4; **Huge** went from 46.7 to 52.1.

**Why it matters.** Records in every size group keep rising, which means a model's date matters as much as its size: a recent small model can beat an older large one. When choosing, prefer newer releases at the same size.

### Figure 24 · How far behind are the small models?

![Figure 24](figures/fig24_small_catches_big.png)

**What this graph shows.** Two record lines over time: the best model that fits a 4 GB device, and the best model of 30 billion parameters or more.

**How to read it.** The black bar joins the moment the giants first reached today's best small-model score to the moment the small models reached it. Its length is how far small models lag behind.

**What the data says.** The best model that runs on 4 GB now scores **35.3**. Models of 30B or more first reached that score in **July 2023**; the small models got there in **December 2024** — a gap of about **17 months**, for a model roughly 5× smaller.

**Why it matters.** Capability flows downhill: what needs a big server today runs on a phone-class device some months later. For anyone building for weak hardware, the right question is not 'can a small model do this?' but 'how long until one can?'.

## Part 6 · The decision: what should you run on 4, 8 or 16 GB?

Everything so far turns into a choice. From here on we use only models that are **still downloadable**, and we size them at 4-bit: a model fits 4 GB if it has at most 5.8B parameters, 8 GB → 11.6B, 16 GB → 23.3B.

### Figure 25 · Which sizes give the most capability per GB?

![Figure 25](figures/fig25_score_per_gb.png)

**What this graph shows.** How many score points each model delivers for every gigabyte of memory it needs.

**How to read it.** Higher = more capability per GB. The vertical axis is stretched so each step is ×10. Boxes cover the middle half of each group; the number is the middle model.

**What the data says.** Middle model's points per GB: **Tiny 25.5**, **Small 11.5**, **Medium 5.3**, **Large 4.3**, **X-Large 2.0**, **Huge 0.9**. The smallest group delivers about **27×** more score per GB than the biggest.

**Why it matters.** Every extra gigabyte buys less than the one before — the returns shrink. This doesn't mean small is better (the total score still matters); it means each step up in memory must be justified by a real need.

### Figure 26 · The efficiency frontier

![Figure 26](figures/fig26_frontier.png)

**What this graph shows.** Every downloadable model's memory need (left to right) against its score. The orange line is the **frontier**: the models that no smaller model can beat. Everything below the line is beaten by something that needs the same or less memory.

**How to read it.** Find your device's blue line; the highest orange point to its left is the best you can run. The flatter the orange line gets, the less each extra GB buys.

**What the data says.** Best model per budget: **4 GB → 3Bgeneralv2-ECE-PRYMMAL-Martial** (3.0B, score 31.5); **8 GB → Chocolatine-Fusion-14B** (8.4B, score 40.4); **16 GB → T3Q-qwen2.5-14b-v1.0-e3** (15B, score 47.1). 4 → 8 GB adds **+8.9** points; 8 → 16 GB adds **+6.7** points.

**Why it matters.** A model below the frontier is never the right choice — something smaller is at least as good. That alone rules out most of the leaderboard for any given device.

### Figure 27 · THE PAYOFF: old rule against new rule

![Figure 27](figures/fig27_payoff_old_rule_vs_new.png)

**What this graph shows.** For each device size, three choices side by side. Grey: what you typically get if you grab one of the biggest models that fit. Aqua: the best model from an official lab that fits. Blue: the best model of any kind that fits.

**How to read it.** Compare the grey bar with the blue bar in each group — the blue number above is the gain from choosing by score instead of by size.

**What the data says.** On **4 GB**, a typical biggest-that-fits model (3.6B–5.5B) scores **21.5**, while the best that fits (**3Bgeneralv2-ECE-PRYMMAL-Martial**, 3.0B) scores **31.5** — **+10.0** points, equal to a model about 5× bigger. On **8 GB**, a typical biggest-that-fits model (8.8B–12B) scores **23.2**, while the best that fits (**Chocolatine-Fusion-14B**, 8.4B) scores **40.4** — **+17.2** points, equal to a model about 18× bigger. On **16 GB**, a typical biggest-that-fits model (15B–23B) scores **38.2**, while the best that fits (**T3Q-qwen2.5-14b-v1.0-e3**, 15B) scores **47.1** — **+8.9** points, equal to a model about 4× bigger.

**Why it matters.** This is the answer to the project's question. The rule 'biggest that fits' leaves a large amount of capability on the table; 'best score that fits' recovers it at no extra hardware cost. The aqua bar is the safe option: official releases are less likely to be tuned just to score well on this leaderboard.

### Figure 28 · The shortlist for each device

![Figure 28](figures/fig28_top_picks.png)

**What this graph shows.** The eight best-scoring downloadable models that fit each device at 4-bit.

**How to read it.** Longer bar = higher score. Bar colour shows how the model was made; a star marks an official lab release. Read top to bottom from best to eighth best.

**What the data says.** On **4 GB** the top model is **3Bgeneralv2-ECE-PRYMMAL-Martial** (3.0B, 31.5). On **8 GB** the top model is **Chocolatine-Fusion-14B** (8.4B, 40.4). On **16 GB** the top model is **T3Q-qwen2.5-14b-v1.0-e3** (15B, 47.1).

**Why it matters.** This is the practical output: a shortlist to test on your own task. Because of Figure 22, check the individual test that matches your job before picking from the top two or three.

### Figure 29 · What does size cost in energy?

![Figure 29](figures/fig29_co2_cost.png)

**What this graph shows.** The estimated CO₂ produced by running the six tests on each model, against the model's size. The energy to test a model is a good stand-in for the energy (and money) needed to run it.

**How to read it.** Both axes are stretched so each step is ×10. A straight trend on this kind of chart means cost grows by a fixed multiple each time size grows ×10.

**What the data says.** Each time a model gets 10× bigger, the energy to run it grows about **3×**, while its score grows by only **13.8 points**.

**Why it matters.** This is the hidden cost of 'bigger is better': size buys points slowly but costs energy fast. It is the strongest argument for picking the smallest model that does the job — cheaper, faster, and greener.

### Figure 30 · Are popular models the best ones?

![Figure 30](figures/fig30_popularity_vs_score.png)

**What this graph shows.** How many likes each downloadable model has on Hugging Face, against its leaderboard score.

**How to read it.** Dots to the right score higher; dots higher up are more popular. If popularity followed quality, the dots would rise from bottom-left to top-right.

**What the data says.** The rank agreement between likes and score is **0.09** (1 = perfect, 0 = none) — **almost no** link. The middle official release has **148** likes; the middle community model has **1**, whatever its score.

**Why it matters.** Picking by popularity or brand name is another weak rule. Popular models are a sensible safe start, but the best model for a budget is often a less famous one — Figures 26–28 show which.

### Figure ★ · Interactive explorer

Interactive file: `interactive_explorer.html`

**What this graph shows.** The main chart again, interactive: hover over any dot to see the model's name, size, memory need and how far it beats its size.

**How to read it.** Click a model type in the legend to hide or show it; drag to zoom; double-click to reset.

**What the data says.** Use it to find specific models mentioned in this report, or to explore a size range you care about.

**Why it matters.** Charts summarise; this lets you check any individual model yourself.
