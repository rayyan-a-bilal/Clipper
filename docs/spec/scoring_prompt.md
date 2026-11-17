# LLM Scoring Rubric

## Rubric Axes (0-10 scale)
1. **Hook** - How well does this clip hook the viewer in first 2.5 seconds?
2. **Payoff** - Does this clip deliver on its promise? Is it satisfying/engaging?
3. **Self-contained** - Does this clip stand alone or does it require context from elsewhere?
4. **Emotion** - How emotionally engaging is the clip? 
5. **Clarity** - Is this clip easy to understand without audio?

## Scoring Guidelines
For each of the 5 axes, score on a scale of 0-10:
- 0-2: Very poor - completely fails this criterion  
- 3-4: Poor - significantly fails this criterion
- 5-6: Average - meets basic expectations
- 7-8: Good - exceeds expectations 
- 9-10: Excellent - exceptional in this area

## Input Format for Scoring LLM
```
Clip content:
[clip content description]

Context:
[relevant sentence or part of transcript]

Scoring Guidelines:
[insert the rubric axes and guidelines above]
```

## Output Format
```json
{
    "hook": integer between 0-10,
    "payoff": integer between 0-10,
    "self_contained": integer between 0-10,
    "emotion": integer between 0-10,
    "clarity": integer between 0-10
}
```

## Constraints  
- Must return valid JSON
- All scores must be integers between 0-10 inclusive
- Response must follow the exact JSON schema above