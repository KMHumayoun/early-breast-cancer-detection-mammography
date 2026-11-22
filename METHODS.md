# Study protocol

## Screening question

The unit of prediction is one breast in one screening exam. The outcome is the RSNA cancer field. The primary comparison is CC + MLO against MLO alone, evaluated on the same breasts at the second site. The development site, two models, image size, epoch budget, selection metric and cutoff rule are specified here before inspecting any patient-level results.

## Cohort

Start from the official labeled `train.csv` and `train_images` directory. Group images by patient and laterality. The RSNA release supplies one screening examination per participant for this design. A patient in two sites, duplicate image ID, nonbinary outcome, missing selected file or conflicting label within a breast stops preparation.

Keep breasts with both an exact CC and an exact MLO view, and use the same eligible breasts in both arms. Additional views have no effect on selection. If a standard view was repeated, choose the lowest numeric image ID for that view, independent of its pixels and label. The preparation audit records exclusions by cancer status and counts extra standard views. The paired-view eligibility rule can change the case mix. An image ID check does not detect duplicated pixel content under different IDs.

Keep the sites separate. Within site 1, allocate patients in approximate proportions 70/15/15, stratified by whether either eligible breast has cancer. Site 2 supplies the test group. A breast from the same patient can never occur on both sides of the split. The index stores the source CSV hash. The evaluation also checks that the models were trained on the exact same index.

## Images and models

Read the grayscale DICOM pixel array with its transfer syntax decoder. Mask DICOM padding and account for MONOCHROME1 polarity. Use the 0.5th and 99.8th sampled percentiles to scale the display range, keeping the original aspect ratio when fitting the image inside a 768 × 512 canvas. The image is converted to eight-bit display intensity for resizing. Small calcifications may be lost at this resolution. No reader assessment, biopsy flag, density field, invasive status, machine ID or age enters the model.

Both arms fine-tune an ImageNet-initialized ResNet-18. Its final convolutional feature map feeds a learned spatial weighting and a pooled descriptor. The paired model concatenates the descriptors in CC, MLO order. Both use dropout 0.2 and a binary output. The extra input changes the size of the final head. Attention weights have no verified lesion-location meaning.

Use seed 83, a batch of two and ten epochs. AdamW uses learning rate 0.0001 and weight decay 0.0001. The positive loss weight is estimated on training breasts as negatives divided by positives. Model selection uses the highest average precision on selection patients, with the earliest epoch winning ties. Training loss weights change the scale of the raw outputs.

## Calibration and independent-site testing

On the calibration split, fit a univariate logistic regression to each model’s logit. Set the decision cutoff just above the negative-score order statistic that yields at least 90% empirical specificity. Ties are handled conservatively. Apply that same calibration and cutoff at the test site. Sensitivity and specificity there may differ from those at development.

Report test patient, breast and cancer counts and prevalence. The primary effect is the difference in breast-level average precision. Resample whole patients with replacement 1,000 times and evaluate both models on exactly the same draw. Report percentile intervals and the number of valid draws. Draws without both classes cannot give ROC AUC and are omitted. Intervals condition on these fitted models, this split and these patients. They do not capture training-seed variability.

Compare either model’s precision–recall curve with the sampled prevalence. The held-out site differs in population, acquisition equipment and case selection. The study cannot attribute differences to any one of these factors. The enriched dataset also limits interpretation of precision, calibration and the fraction flagged in routine screening. Interval cancers are outside the dataset. Exam-level stage and lesion localization remain outside this endpoint.

## Reference

[RSNA dataset paper and outcome definitions](https://pubs.rsna.org/doi/10.1148/ryai.250375) · [Official field descriptions and access terms](https://github.com/RSNA/AI-Challenge-Data/wiki/RSNA-Screening-Mammography-Breast-Cancer-Detection)
