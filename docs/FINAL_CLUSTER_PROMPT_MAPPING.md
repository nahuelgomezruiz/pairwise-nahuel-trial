# Final Cluster-to-Prompt Mapping Based on Data Analysis

## Overview

After extensive analysis of the actual essay content found by K-means clustering, we have defined **8 new prompts** that accurately reflect what the algorithm discovered. These prompts are based on comprehensive keyword analysis, writing style analysis, and examination of hundreds of sample essays.

## The Data-Driven Approach

Instead of forcing the original 8 prompts from `prompt_notes.txt` onto arbitrary clusters, we:

1. **Analyzed actual cluster content** - examined keywords, writing styles, and sample essays
2. **Identified natural topic groupings** - found what the algorithm actually clustered together  
3. **Created accurate prompts** - wrote prompts that match the real content
4. **Used OR conditions** - allowed for merged prompt types where clusters contained multiple styles

## Final Cluster Mapping

### Cluster 0: Car Usage Limitation Benefits
- **Essays**: 314
- **Topic**: Advantages of limiting/reducing car usage in communities
- **Key Vocabulary**: cars, usage, limiting, advantages, pollution, transportation, Germany, Paris, Vauban, smog
- **Style**: Argumentative, environmental focus
- **Examples**: German car-free suburbs, Paris driving bans, pollution reduction

**Prompt**: *"Discuss the advantages and benefits of limiting car usage in communities"*

---

### Cluster 1: Face on Mars Analysis  
- **Essays**: 379
- **Topic**: Analysis of whether Mars "face" is alien-created or natural
- **Key Vocabulary**: face, mars, aliens, landform, natural, NASA, created, Viking, mesa, picture, believe
- **Style**: Mixed argumentative and evidence-based
- **Includes**: Both basic claim evaluation AND evidence-based analysis

**Prompt**: *"Evaluate whether the 'Face on Mars' is evidence of aliens OR a natural landform using scientific analysis"*

---

### Cluster 2: Venus Exploration
- **Essays**: 565  
- **Topic**: Whether Venus exploration is scientifically worthwhile
- **Key Vocabulary**: venus, planet, author, earth, dangers, exploring, studying, worthy, pursuit
- **Style**: Argumentative with evidence-based reasoning
- **Focus**: Scientific merit, challenges vs benefits

**Prompt**: *"Is studying/exploring Venus a worthy scientific pursuit?"*

---

### Cluster 3: Driverless Cars
- **Essays**: 609
- **Topic**: Arguments for/against autonomous vehicles
- **Key Vocabulary**: cars, driverless, drive, driving, technology, future, safe, accidents, human
- **Style**: Argumentative, technology-focused
- **Focus**: Safety, technology reliability, human vs machine control

**Prompt**: *"Argue for or against allowing driverless cars on public roads"*

---

### Cluster 4: Emotion Recognition in Schools (FACS)
- **Essays**: 214
- **Topic**: Use of emotion recognition technology in educational settings
- **Key Vocabulary**: technology, students, help, emotions, classroom, computer, facial, valuable
- **Style**: Argumentative with educational focus
- **Focus**: Benefits/concerns for students and learning

**Prompt**: *"Should schools use emotion-recognition (FACS) technology?"*

---

### Cluster 5: Electoral College
- **Essays**: 345
- **Topic**: Debate over keeping or abolishing the Electoral College
- **Key Vocabulary**: electoral, college, vote, president, popular, electors, election, system
- **Style**: Argumentative, civics-focused
- **Focus**: Democratic representation, fairness, system effectiveness

**Prompt**: *"Should the U.S. keep or abolish the Electoral College?"*

---

### Cluster 6: Seagoing Cowboys Program
- **Essays**: 293
- **Topic**: Mixed persuasive and narrative essays about the program
- **Key Vocabulary**: seagoing, cowboys, program, luke, join, help, animals, world, places
- **Style**: **Mixed** - predominantly persuasive (74 markers) with narrative elements (16 markers)
- **Includes**: BOTH persuasive appeals to join AND personal narratives

**Prompt**: *"Persuade someone to join the Seagoing Cowboys program OR provide a narrative/personal account of being a seagoing cowboy"*

---

### Cluster 7: Facial Action Coding System Technology
- **Essays**: 281
- **Topic**: Technical discussion of FACS and emotion analysis
- **Key Vocabulary**: facial, emotions, technology, system, coding, action, computer, Mona Lisa
- **Style**: Technical/analytical, often references Mona Lisa analysis
- **Focus**: How the technology works, emotion recognition capabilities

**Prompt**: *"Discuss the Facial Action Coding System (FACS) technology and its applications in analyzing emotions"*

---

## Key Insights

### 1. **Natural Topic Separation**
The clustering successfully separated:
- **Car limitation** (Cluster 0) vs **Driverless cars** (Cluster 3)
- **FACS in schools** (Cluster 4) vs **FACS technology analysis** (Cluster 7)
- **Venus exploration** (distinct and well-defined)
- **Electoral College** (distinct and well-defined)

### 2. **Mixed Prompt Types**
Some clusters contain **multiple prompt styles**:
- **Cluster 1**: Both basic Mars face analysis AND evidence-based evaluation
- **Cluster 6**: Both persuasive appeals AND personal narratives about Seagoing Cowboys

### 3. **Missing Original Prompts**
Two prompts from `prompt_notes.txt` don't have their own clusters:
- **Face on Mars - Claim Evaluation II** (merged into Cluster 1)
- **Seagoing Cowboys - Narrative** (merged into Cluster 6)

This is because K-means clustering found these to be **semantically too similar** to separate from their related prompts.

## Grading Approach

Each prompt uses **OR conditions** where appropriate:
- If a student succeeds at **either** part of a merged prompt, that counts as success
- Grading criteria include vocabulary and indicators for **all valid approaches**
- Students are not penalized for taking one approach over another in mixed clusters

## Technical Notes

- **Clustering Algorithm**: K-means with k=8
- **Embedding Model**: sentence-transformers 'all-MiniLM-L6-v2'  
- **Training Data**: 3,000 randomly sampled essays from train.csv
- **Validation**: Extensive keyword analysis and manual sample review

This mapping reflects the **actual semantic groupings** discovered by the machine learning algorithm rather than predetermined human categories.