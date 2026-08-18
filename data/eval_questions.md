# Evaluation questions

A small hand-written set used to sanity-check retrieval and grounding after
changing chunk size, top_k, or the prompt. Run each against `/ask` and check the
answer *and* the citations.

## Answerable from the corpus

| # | Question | Expected source |
|---|----------|-----------------|
| 1 | How long do I have to submit an expense claim? | expense-policy.md |
| 2 | Can I expense alcohol during client dinners? | expense-policy.md |
| 3 | Is SMS acceptable as a second factor for production? | security-handbook.md |
| 4 | How quickly must I report a suspected breach? | security-handbook.md |
| 5 | How many annual leave days carry over, and when do they lapse? | onboarding-faq.md |
| 6 | What are the core working hours? | onboarding-faq.md |

## Should be refused (not in the corpus)

| # | Question |
|---|----------|
| 7 | What is the capital of France? |
| 8 | What is our parental leave policy? |
| 9 | Who is the CEO of Northwind Labs? |

Questions 7-9 must return "I couldn't find that in the indexed documents."
Question 7 is the direct test of whether grounding is working: an ungrounded
model answers it happily.
