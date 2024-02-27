from pipelines import pipeline
import os
cwd = os.getcwd()
print(cwd)
model_ans_ext = "saved-models/t5-small-ans-ext"
model_qg= "saved-models/t5-small-qg"

nlp = pipeline("question-generation", #"multitask-qa-qg", 
               model=model_qg, 
               tokenizer=model_qg, 
               ans_model=model_ans_ext,
               ans_tokenizer=model_ans_ext,
               qg_format='highlight')


context = """
 The 49ers featured one of the best running games in the NFL in 1976 NFL season. Delvin Williams emerged as an elite back, gaining over 1,200 yards rushing and made the Pro Bowl. Wilbur Jackson also enjoyed a resurgence, rushing for 792 yards. Once again Gene Washington was the teams leading receiver with 457 yards receiving and six scores. The 49ers started the season 6\u20131 for their best start since 1970. Most of the wins were against second-tier teams, although the 49ers did shut out the Rams 16\u20130, in 1976 Los Angeles Rams season on Monday Night Football. In that game the 49ers recorded 10 sacks, including 6 by Tommy Hart. However, the 49ers lost four games in a row, including two against divisional rivals Los Angeles and 1976 Atlanta Falcons season that proved fatal to their playoff hopes. Louis G. Spadia retired from the 49ers in 1977 upon the teams sale to the DeBartolo Family. The team was sold to Edward J. DeBartolo Jr. in March 1977, and despite finishing the season with a winning record of 8\u20136, Clark was fired after just one season by newly hired general manager Joe Thomas (American football executive), who oversaw the worst stretch of football in the teams history.
"""

qas = nlp(context)
print(qas)