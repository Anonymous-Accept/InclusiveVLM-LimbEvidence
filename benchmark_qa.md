# Questions about InclusiveVLM-LimbEvidence

This Q&A explains the design, results, and intended use of InclusiveVLM-LimbEvidence. Section and table references point to the accompanying manuscript.

## Purpose and evaluation design

### 1. What is the main contribution of InclusiveVLM-LimbEvidence?

InclusiveVLM-LimbEvidence makes specific visual errors measurable for individuals with limb deficiencies. It combines new evidence annotations with separate evaluations of detection, body-region attribution, and compatibility-style structured decisions. The main tables cover 21 model variants from six general-purpose vision-language model (VLM) families. The highest model Constrained Attribution score is 29.4% set mIoU, a label-overlap measure, compared with a human reference of 83.4%. Low complete-set recovery also appears across families. Paired output formats, inference-style comparisons, a medical-domain control, and component diagnostics help explain these results. This connects population coverage to concrete questions in visual grounding, spatial assignment, and set prediction, giving researchers defined targets for improving visual AI for individuals with limb deficiencies.

*Paper: Sections 1 and 4, Tables 1, 2, 4, 5, and 6, Appendix D.1.*

### 2. How does this resource differ from LDPose and other inclusive vision benchmarks?

We reuse LDPose images and add benchmark-specific masks for residual limbs and prosthetic devices, body-region labels, body-device records, prompts, and evaluation targets. LDPose primarily evaluates pose estimation, while ProGait supports video-based perception and gait analysis. Counterfactual depiction studies examine how model responses change between depictions. InclusiveVLM-LimbEvidence instead checks whether a language-based answer agrees with the visible limb evidence and target for the evaluated image. These resources address complementary questions about body representation, response variation, and image-specific correctness.

*Paper: Section 2.1, Section 3.3, Appendices A.1 and A.2.*

### 3. Why separate Presence, Attribution, and Compatibility?

Each requirement asks for information that the preceding description alone cannot supply. LimbEvidence-Presence asks which evidence types are visible. Detecting a prosthetic device does not identify the body region or side to which it belongs, which motivates LimbEvidence-Attribution. Identifying that region still does not establish whether a model can match the visible configuration to candidate devices, which motivates LimbEvidence-Compatibility. Constrained and FreeForm are parallel Attribution formats. Category and Diversity are parallel Compatibility probes. The tasks are evaluated separately, so the progression describes additional requirements rather than a pipeline that passes one model answer into the next task.

*Paper: Section 3.1, Figure 1, Appendix B.1.*

### 4. Who defined the probes, and what professional input informed the annotations?

The authors defined the probes from observed errors in VLM prototypes for preparing visual evidence in Paralympic athlete classification workflows. The evaluation targets were fixed before the new annotations were created. An International Paralympic Committee certified classifier supported annotation guidance and case adjudication. This contribution concerns the interpretation of visual evidence within the annotation process. Probe selection remained an author responsibility. The project did not conduct formal participatory co-design, and the manuscript distinguishes these roles rather than describing the benchmark as a complete clinical capability taxonomy.

*Paper: Section 3.1, Section 3.3, Appendices B.1 and B.4.*

### 5. What do limb evidence and Attribution mean here?

Limb evidence means visible residual-limb or prosthetic-device regions and their annotated relation to the body. Attribution assigns each evidence type to a body region and side, such as left-thigh prosthetic-device evidence. The vocabulary covers upper arm, forearm, thigh, and calf on each side, crossed with the two evidence types. Left and right refer to the depicted individual. Residual-limb and prosthetic-device masks can occupy different pixels within the same segment. Models predict labels rather than mask coordinates. Unseen anatomy and intact-limb labels are outside the annotation target.

*Paper: Section 3.3, Appendices B.3 and B.5.*

### 6. What do the dataset counts represent?

The benchmark contains 16,850 source images. Presence and Constrained Attribution each contain 26,406 prompts over 8,802 images, giving 52,812 direct evidence queries. Category contains 8,048 instances and Diversity contains 12,616 instances over 8,048 images, giving 20,664 Compatibility instances. FreeForm is a separate diagnostic on 500 paired image-target items. These are counts of images and evaluation records, not distinct people. Filtering removes near-duplicates, but image counts do not establish person-disjoint evaluation or demographic representativeness. The results describe the evaluated open-world image collection.

*Paper: Section 3.2, Table 3, Appendices B.2 and D.3.*

## Matching relations and human reference

### 7. What determines whether a candidate is compatible?

Compatibility follows a fixed correspondence between the body-part categories annotated for the image and those represented by the candidate devices. For example, an upper-limb device is excluded when the image has only lower-limb target categories. Different device appearances can satisfy the same body-part relation. The visual task is to identify the relevant categories in the image and candidates, then apply that relation. Accepted categories and option sets are recorded and reviewed during annotation. This is structured visual matching, not a decision about which device should be prescribed to an individual.

*Paper: Section 3.4, Appendices B.6 and D.4.*

### 8. How do Category and Diversity handle multiple matching answers?

Category asks for one category. A prediction is correct when it belongs to the accepted category set, even when several categories are acceptable. Diversity asks for the complete matching option set, which can be empty. Its primary F1 describes set overlap, while Exact requires every target and no distractors. Partial records at least one target hit. These probes use separate grids and evaluate different output requirements. Diversity refers to variation in candidate appearance, not demographic diversity or a later step that receives the Category prediction.

*Paper: Section 3.4, Figure 2, Appendix B.6.*

### 9. How are the reference annotations established and reviewed?

Annotators use visual guidelines for residual-limb evidence, prosthetic-device appearance, and body regions. A second annotator reviews each mask and derived label set, including accepted categories and matching option sets. Disagreements above the annotation tolerance lead to re-annotation and adjudication. A certified classifier supports guidance and case adjudication, and periodic spot audits inspect randomly sampled batches. The reviewed masks, labels, and option sets form the reference targets used for model evaluation and human scoring. This makes the connection between the image annotation and the scored answer explicit.

*Paper: Section 3.3, Figure 3, Appendix B.4.*

### 10. How should the human reference results be read across tasks?

The human results describe performance under different response requirements. Raters reach 98.0% Presence accuracy and 83.4% Constrained Attribution mean set intersection over union (set mIoU). For Diversity, the overall F1 is 57.9%, while Exact is 12.0%. Exact gives no partial credit when a required option is omitted or a distractor is added. It therefore measures a different requirement from recovering much of the target set. The human reference provides task-specific comparisons against reviewed targets. The background-specific results below add context that the overall average alone does not show.

*Paper: Tables 1, 2, and 4(a), Appendix C.3.*

### 11. Why report participant background rather than only an overall human average?

The two participants with extensive disability-research experience achieve Diversity F1 scores of 77.78% and 70.67%. Their mean is 74.22%, compared with 47.03% for the participants with primarily computer-science backgrounds. The ratio is 1.58, with a 95% interval of [1.16, 2.22]. An analogy is bird identification, where people with different experience can perform differently under the same answer rules. A mean below the maximum does not by itself identify a problem with those rules. The ratio compares performance, not inter-rater agreement. The comparison is descriptive because assignments were not identical across participants. It is not a causal estimate of experience or a clinical performance ceiling.

*Paper: Table 4(a), Appendix C.3.*

### 12. How should the reported metrics and confidence intervals be interpreted?

Different metrics expose different errors. GPT-5 reaches 77.2% Presence accuracy but 50.0% macro-F1, while Gemini-2.5 reaches 47.0% and 35.3%, respectively. Accuracy weights examples equally, whereas macro-F1 weights the evaluated labels equally. For set outputs, Exact requires complete agreement, while set mIoU and F1 describe overlap. Sample-average and micro-F1 also use different aggregation rules. The tables report 95% confidence intervals where available, with the resampling units specified per evaluation in Appendix C.2. These distinctions help readers assess category imbalance, missed targets, and incorrect inclusions without treating one metric as a complete account of performance.

*Paper: Tables 1 and 2, Appendix C.2.*

## What the experiments establish

### 13. How do the analyses distinguish the demands contributing to model performance?

The evidence comes from several complementary comparisons, not one model. The main tables compare six VLM families. The paired Attribution study holds images and targets fixed while changing the output format. Six within-family inference-style pairs and the matched MedGemma comparison examine checkpoint style and specialization. Figure 4 shows both correct and incorrect Presence answers across families, plus a Category example where every displayed prediction is incorrect. The component analysis provides a focused view of spatial assignment. These analyses support task-level conclusions about observable model outputs without attributing every error to one mechanism. They make the findings relevant to model evaluation while keeping the role of vocabulary, format, and familiarity explicit.

*Paper: Tables 1, 2, 4(b), 5, and 6(c), Figure 4, Appendices C.4 through C.7 and D.2.*

### 14. Does a much larger model resolve the observed failures?

Not uniformly. Moving from Qwen3-VL-30B-A3B-Instruct to 235B-A22B-Instruct raises Presence accuracy from 23.9% to 74.23%, but Constrained Attribution set mIoU changes only from 24.0% to 24.76%. Diversity F1 falls from 51.2% to 25.90%. A task-dependent pattern also appears within Gemma-3. Its reported Constrained Attribution set mIoU rises from 20.5% at 4B to 27.8% at 27B, while Category accuracy falls from 16.3% to 10.5%. Larger checkpoints can improve some requirements without improving others. These are comparisons of the evaluated checkpoints and settings, not a controlled estimate of parameter count alone or a general scaling law.

*Paper: Section 4.1, Tables 1, 2, and 5(a), Appendix C.4.*

### 15. Why include Thinking and Reasoning checkpoints for visual tasks?

They test whether the findings depend on the evaluated inference style. Across six same-family, same-scale pairs, no Thinking or Reasoning checkpoint improves all four primary probe scores. Qwen3-VL-4B-Thinking improves Diversity F1 by 17.4 percentage points while reducing Category accuracy by 11.8 points. For Ministral-3, Reasoning increases Diversity F1 by 6.9, 7.9, and 13.9 points at 3B, 8B, and 14B, respectively, but Exact decreases or remains unchanged at the reported precision. Including these results makes the trade-offs visible instead of assuming that reasoning either always helps or never helps. The comparisons concern existing checkpoints, not an isolated intervention on reasoning.

*Paper: Table 5(b), Appendix C.4.*

### 16. Does medical-domain specialization improve these results?

The matched MedGemma 1.5 4B comparison does not show improvements on the three reported Presence, Attribution, and Diversity primary scores. Relative to Gemma-3 4B, the differences are -7.87 points in Presence accuracy, -12.31 in Attribution set mIoU, and -4.56 in Diversity F1. The Category difference has magnitude 1.12 points, with a paired interval that includes zero. Medical-domain specialization is therefore not sufficient for better performance in this comparison. The conclusion concerns the evaluated model and open-world limb evidence, not all medical VLMs or clinical capabilities.

*Paper: Table 5(c), Appendix C.4.*

### 17. Why does FreeForm use 500 paired items, and what does the comparison show?

FreeForm is an output-format diagnostic for an attribution target already evaluated at full scale in Constrained form. We randomly sampled 500 image-target pairs and held both the image and target fixed within each paired comparison. On this subset, Qwen3-VL-4B-Thinking changes from 17.22% Constrained set mIoU to 43.1% FreeForm set mIoU. The corresponding 8B values are 15.87% and 44.7%, while GPT-5 changes from 26.51% to 30.5%. Output format clearly affects measured performance. The comparison is not an assessment of unrestricted description quality, and its Constrained scores should not be replaced with the full-pool scores.

*Paper: Section 4.1, Table 4(b), Appendices B.5 and C.5.*

### 18. What does FreeForm parsing coverage establish?

Parsing coverage measures whether a deterministic parser can recover a canonical label set, not whether that set is correct. In the paired diagnostic, coverage is 93.4% for Qwen3-VL-4B-Thinking, 90.4% for 8B-Thinking, and 83.4% for GPT-5. The remaining outputs are invalid under the parser. Attribution is then evaluated against the benchmark targets. Reporting coverage alongside set mIoU helps distinguish recoverable output structure from correct evidence assignment. The observed format differences include the effects of generation and deterministic normalization, rather than showing that the output interface is irrelevant.

*Paper: Table 4(b), Appendix C.5.*

### 19. What does low exact-set accuracy reveal beyond task difficulty?

Low Exact is not confined to one model family. Qwen3-VL-30B-A3B-Instruct reaches 80.4% Partial but 7.6% Exact. Gemma-3-27B reaches 56.0% and 0.9%, and Gemini-2.5 reaches 63.5% and 3.5%, respectively. Precision and recall reveal different error profiles. DeepSeek-VL2-Tiny combines 81.3% recall with 49.6% precision, while GPT-5 has 18.9% recall and 53.3% precision. High target coverage with incorrect inclusions and low target coverage are different failures, even when both produce low Exact. Table 7 further separates missed targets, extra options, and no-match errors in its detailed case analysis. This makes set prediction more informative than a single count of imperfect answers.

*Paper: Section 4.2, Tables 2 and 7, Appendix C.7.*

### 20. How are simple image-independent response strategies used to interpret the scores?

Appendix C.9 uses target counts to show what a response can achieve without inspecting the image. In the Presence pool analyzed there, always choosing the most frequent label yields 65.79% accuracy. In the corresponding 12,616-item Diversity pool, always returning an empty set yields 14.67% Exact, while selecting every option yields 85.33% Partial. These references show why a high Partial score or a nonzero Exact score alone need not demonstrate image-grounded matching. We therefore report overlap, selection size, and empty-target outcomes alongside Exact and Partial. The references are analytical calculations, not additional model runs, and comparisons require the same evaluation pool.

*Paper: Appendix C.9, Tables 6(b) and 7.*

### 21. What does paraphrase consistency add beyond correctness?

Correctness checks agreement with the target, while consistency checks variation across equivalent wording. The main tables show why they are not interchangeable. Gemma-3-12B has 85.2% gated Presence consistency but 29.9% accuracy. DeepSeek-VL2-Tiny has 87.0% gated Diversity consistency but 10.2% Exact. Gated consistency applies only to groups with at least one correct response, so these values do not describe overall correctness. The separate raw-consistency analysis supplies direct evidence of repeated errors: Qwen3-VL-235B gives the same incorrect Attribution answer throughout 47.28% of paraphrase groups. We therefore interpret correctness, raw consistency, gated consistency, and retained coverage separately. A stable answer can be wrong, and high consistency on a retained subset is not a measure of success across all examples.

*Paper: Tables 1, 2, and 6(e), Section 4.3, Appendix C.8.*

### 22. How are no-match answers, insufficient information, and invalid outputs distinguished?

A no-match target means none of the presented candidates satisfies the benchmark relation. An insufficient-information choice, where provided, is an explicit answer indicating inadequate visible evidence. An invalid output is one from which the parser cannot recover a valid canonical response. These outcomes have different meanings. Correctly returning an empty set does not by itself establish uncertainty-based abstention, and a parsing failure is not a successful decision to withhold an answer. Keeping the cases separate makes the selection errors and the scope of the reported behavior easier to interpret.

*Paper: Appendix B.6, Appendix C.7.*

### 23. How can a test-only benchmark guide future model development?

The experiments identify several concrete development targets. Low Attribution scores across model families motivate better evidence-to-region assignment. Diversity precision and recall distinguish missed targets from incorrect inclusions, while the paired FreeForm results show why improvements should be checked across output formats. The 235B oracle analysis adds a focused diagnostic: correcting body region or side separately raises set mIoU by 16.32 or 21.41 percentage points. These are target-assisted gains, not results from a trained remedy. The exact pretraining corpora are unavailable, so the study does not estimate population-specific training prevalence. Future methods should use separate development data and test whether they address these observed errors on the fixed benchmark, rather than improve only coarse accuracy or response consistency.

*Paper: Tables 1, 2, 4(b), and 6(c), Appendices B.7, C.6, D.2, and D.4.*

## Access and use

### 24. How will reproduction remain traceable when images, models, or device pools change?

The release plan pairs benchmark annotations and task records with stable image identifiers, checksums, exact option grids, scoring code, and reconstruction instructions. Source-image access follows the permitted LDPose access route rather than relying only on changing web links. Recorded model identifiers and API snapshot dates will accompany closed-model results. Candidate-pool updates will use new versions and change logs, preserving the relationship between prior scores and their original targets. These plans distinguish reproducing target-based scoring from rerunning a provider service that may change. We will document the non-owner download and scoring check alongside the release.

*Paper: Appendix C.1, Appendices E.2 and E.3.*

### 25. What ethical safeguards govern the benchmark?

The project received approval from an institutional Human Research Ethics Committee. Institution names and approval identifiers are omitted during anonymous review. Benchmark annotations, prompts, and derived metadata are designated for CC BY-NC-SA 4.0 release, while source photographs retain their own rights and access conditions. The access agreement restricts use to non-commercial research, requires secure storage, and prohibits re-identification and face recognition. Correction and removal procedures support affected individuals and rights holders. The resource evaluates model interpretation of visible evidence, not the abilities, identities, or lived experiences of the people depicted.

*Paper: Ethics statement, Appendices E.1 and E.2, Datasheet.*

## About this Q&A

### 26. Why include a Q&A section on the project page?

At some point, we noticed we were answering imaginary reviewers in the middle of the Introduction. That was not our best writing habit. So the paper now follows the research story, and this page takes the follow-up questions. You can open the ones you care about and skip the rest. And yes, we hope these answers help earn your support for the paper. We care about making individuals with limb deficiencies part of VLM evaluation, and we hope the benchmark and evidence make a convincing case. Thanks for reading this far.
