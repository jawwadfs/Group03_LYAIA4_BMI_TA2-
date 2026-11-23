# Ethical Reflection: The Limitations of BMI

## What BMI actually measures

Body Mass Index is just weight divided by height squared. It was
developed in the 1830s by a Belgian statistician, Adolphe Quetelet, who
was trying to describe the "average man" across a population — not to
assess any one person's health. That origin matters, because it explains
BMI's biggest weakness: it was built as a population-level statistic and
gets used today as if it were an individual diagnostic tool.

## The main limitation: BMI can't tell fat from muscle

BMI only uses weight and height. It has no way to know whether that
weight is mostly muscle, bone, fat, or water. This creates real
misclassifications. A well-trained athlete or bodybuilder can easily
land in the "Overweight" or even "Obesity" category on paper, purely
because muscle is denser than fat, even though their actual body fat
percentage is low and they're in excellent health. The reverse also
happens: someone with a "normal" BMI can still carry a high percentage
of body fat and be at real metabolic risk — sometimes called being
"skinny fat." BMI simply can't see the difference between these cases.

On top of that, BMI doesn't account for where fat is stored (visceral
fat around organs is riskier than fat elsewhere), and its cutoffs were
largely derived from studies on adult populations, so they don't map
cleanly onto every age group, sex, or ethnicity. Several health
organizations now treat BMI as a rough screening tool to flag who might
need a closer look, not as a stand-alone verdict on someone's health.

## Should the agent mention this limitation? Yes.

If our agent just outputs "BMI: 27.3 | Category: Overweight" and stops
there, it's very easy for a user to read that as "the AI says I'm
unhealthy," which isn't what BMI can actually tell you. That's why the
system prompt explicitly instructs the model to frame BMI as a general
screening measure and to avoid language that sounds like a diagnosis.
Given how casually people already treat numbers from apps and calculators
as authoritative, an AI agent repeating a category without context runs
a real risk of reinforcing an inaccurate or even harmful self-assessment
— especially for people already prone to anxiety about their weight. A
short, honest caveat costs us almost nothing to include, and it makes
the tool meaningfully more responsible.

## Bottom line

BMI is a useful, cheap, easy-to-compute screening number at the
population level. It is not a complete measure of an individual's
health, and this project is designed to say so every time it gives an
answer, rather than presenting a single number as the whole story.
