# -*- coding: utf-8 -*-
"""build_manuscript.py v3  CEA 英文 SCI 手稿
- 摘要<200词；引言含两段真实文献调研；19个公式；结果/讨论机理分析补全；21条真实文献
- 实验数字动态读取 results/*.csv；涉密内部件不写
"""
import os, sys
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
sys.path.insert(0, os.path.dirname(__file__)); from config import PROJ
MT=os.path.join(PROJ,"main_text"); RES=os.path.join(PROJ,"results")

doc=Document()
s0=doc.styles["Normal"]; s0.font.name="Times New Roman"; s0.font.size=Pt(11)
s0.paragraph_format.line_spacing=1.5; s0.paragraph_format.space_after=Pt(4)
def H(t,l=1):
    h=doc.add_heading(t,level=l)
    for r in h.runs: r.font.name="Times New Roman"; r.font.color.rgb=RGBColor(0,0,0)
def P(t,j=True):
    p=doc.add_paragraph(t); p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY if j else WD_ALIGN_PARAGRAPH.LEFT; return p
def CAP(t):
    p=doc.add_paragraph(t)
    for r in p.runs: r.font.size=Pt(10); r.bold=True
def IMG(f,w=6.6):
    doc.add_picture(os.path.join(MT,f),width=Inches(w)); doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
def EQ(e,n):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.tab_stops.add_tab_stop(Inches(6.5),WD_TAB_ALIGNMENT.RIGHT)
    r=p.add_run(e); r.italic=True; r.font.name="Cambria Math"; r.font.size=Pt(11)
    p.add_run(f"\t({n})")
def TBL(df):
    t=doc.add_table(rows=1,cols=df.shape[1]); t.style="Light Grid Accent 1"
    for i,c in enumerate(df.columns): t.rows[0].cells[i].text=str(c)
    for _,row in df.iterrows():
        cc=t.add_row().cells
        for i,v in enumerate(row): cc[i].text=str(v)

# ---------------- Title ----------------
t=doc.add_paragraph(); t.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=t.add_run("Physics-constrained unsupervised operating-condition recognition from production-equivalent signals for energy management pre-processing on a series-hybrid agricultural tractor")
r.bold=True; r.font.size=Pt(15)
a=doc.add_paragraph(); a.alignment=WD_ALIGN_PARAGRAPH.CENTER; a.add_run("[Author 1]a, [Author 2]a,*, [Author 3]b\n")
af=doc.add_paragraph(); af.alignment=WD_ALIGN_PARAGRAPH.CENTER
af.add_run("a [School/College, University, Address, City, China]\nb [YTO Group Corporation / R&D Center, Luoyang, China]\n* Corresponding author: [email]").font.size=Pt(10)

# ---------------- Abstract (<200 words) ----------------
H("Abstract")
P("Recognizing the operating condition is a prerequisite for predictive energy management on hybrid tractors, yet field data lack annotations and the powertrain variables that make labelling easy are consequences rather than causes of the load. We present a physics-constrained, annotation-free framework and demonstrate it on a 220-hp series-hybrid tractor with a four-bottom, 30 cm-deep plough on hard soil in northern Xinjiang. Seven 200 Hz segments were merged into 3200 three-second windows. A hierarchical physically-constrained Gaussian mixture model (PC-GMM) generated five-condition pseudo-labels that agreed with an independent reference at 97%, versus 61-80% for K-means and unconstrained GMM. Under leave-one-segment-out validation, tree ensembles reached 0.973 accuracy and 0.920 macro-F1 using transducer draft force plus production signals, and 0.842/0.764 using only production-equivalent signals; the four steady conditions were recalled at 97-99%, and implement entry/exit transitions were the main residual error. Ablation showed draft force is decisive, wheel slip adds a small gain, and frequency features add none; matched-speed checks and a 0.996 power-balance correlation confirmed physical validity. An INT8 network retained production accuracy in an 11 kB, 81-microsecond model, enabling low-cost embedded deployment.")
P("Keywords: series-hybrid tractor; operating condition recognition; physics-constrained Gaussian mixture model; wheel slip; energy management; INT8 quantization; edge deployment",j=False)

# ---------------- 1 Introduction ----------------
H("1. Introduction")
P("Electrification and hybridization are increasingly applied to agricultural tractors because field loads are highly variable and frequently force the engine away from its efficient region. In a series-hybrid configuration, the engine-generator set is mechanically decoupled from the drive wheels, creating freedom to schedule engine speed and torque independently of the instantaneous traction demand. Realizing this benefit requires a supervisory energy management strategy (EMS) that anticipates the operating condition: idle stops, headland manoeuvres, implement entry/exit transitions, and tillage at different load levels each call for a different power and state-of-charge policy. The operating-condition recognizer therefore sits upstream of the EMS and must deliver its decision before, or at the start of, each condition.")
P("Data-driven recognition of tractor operations has grown rapidly along two lines. The first classifies the broad operation type from on-board signals: random forest and gradient boosting predicted equipment utilization from five SAE J1939 parameters [1]; convolutional and lightweight RepViT networks recognized tractor conditions and mounted implements from speed and engine data or field images [2]; and self-supervised graph-attention and GNSS autoencoder models identified trajectory operation modes [3,4]. A parallel line predicts the tillage load itself, using artificial neural networks for chisel-cultivator draft force and large neural ensembles for off-road tractive force [5]. For hybrid tractors, energy management has been formulated with dynamic programming, model predictive control, equivalent-consumption minimization and deep reinforcement learning, and predicted working conditions have already been embedded in the control loop [6,7]. These studies establish that machine learning can capture tractor operating patterns.")
P("Three gaps nevertheless remain. First, most recognizers distinguish broad operation classes-ploughing versus transport versus headland, or which implement is mounted-rather than the load level within one continuous operation, which is what an EMS needs once a pass has begun, and the dynamic implement entry/exit is rarely treated as a first-class output. Second, accuracy is typically assessed on randomly or stratified splits drawn from adjacent, heavily overlapping windows and is often near perfect, leaving generalization to an unseen pass unverified; these models also commonly take engine speed or torque and motor/generator power as inputs. Because motor power is approximately draft force times speed, such variables are consequences of the load and are not available before the load is applied, and using them both to define and to predict the condition is feature leakage in the sense of Kaufman et al. [8], which inflates accuracy and makes the recognizer unusable as an EMS front end. Third, because field data are unannotated the labels must come from clustering, yet unconstrained clustering is free to partition along physically meaningless axes and is seldom checked against physics, and the embedded cost of the recognizer is rarely quantified.")
P("This work addresses these gaps with an annotation-free, physics-constrained pipeline and a strict separation between the causes available before load is applied and the consequences used only for validation. The contributions are: (1) a hierarchical PC-GMM that builds pseudo-labels along physically ordered variables and isolates the transition class through posterior mixing, reaching 97% agreement with an independent reference without annotation versus 61-80% for K-means and unconstrained GMM; (2) an explicit comparison between production-equivalent signals and research-grade draft-force measurements that quantifies the cost of removing the expensive transducer; (3) a leave-one-segment-out benchmark of six classifiers with statistical significance, progressive ablation of wheel-slip and frequency features, and a principal-component analysis of the mechanisms; and (4) an embedded-deployment study including an engineering feature subset and INT8 quantization with measured size and latency. Powertrain power is withheld from labelling and classification and is used solely for posterior physical validation.")

# ---------------- 2 Materials and methods ----------------
H("2. Materials and methods")
H("2.1 Test platform",2)
P("The test vehicle was an HB2204 wheeled tractor, a 220-hp four-wheel-drive series-hybrid platform with a frameless structure. A YTO LR6M3R turbocharged intercooled diesel engine (162 kW rated at 1900-2000 r/min; Stage IV after-treatment of DOC, DPF and SCR) drives a permanent-magnet synchronous generator rated at 175 kW. A 160 kW permanent-magnet synchronous traction motor (rated/maximum speed 2000/6000 r/min; rated/maximum torque 784/1000 Nm) propels the vehicle through a two-range automated mechanical transmission that is continuously variable within each range (field mode 0-16 km/h). A 14.9 kWh ternary lithium battery (532.8 V nominal) buffers the power difference; the rear hitch is Category 3. Only specification-level parameters relevant to the method are given in Table 1; proprietary internal details are omitted.")
spec=pd.DataFrame([
 ("Diesel engine rated power","162 kW @ 1900-2000 r/min"),
 ("Generator (PMSM) rated power","175 kW"),
 ("Traction motor (PMSM) rated power","160 kW; 784/1000 Nm rated/max"),
 ("Traction battery","14.9 kWh, 532.8 V, ternary Li-ion"),
 ("Transmission","2-range AMT, continuously variable within range"),
 ("Wheelbase","2990 mm"),
 ("Operating mass (approx.)","9200-9350 kg; ballast 1000 front / 400 rear kg"),
 ("Tyres (front/rear)","480/70R30 / 520/85R42; 150-180 kPa field pressure"),
],columns=["Item","Specification"])
CAP("Table 1. HB2204 specification relevant to this study."); TBL(spec)

H("2.2 Field experiment",2)
P("The tractor was coupled to a four-bottom mouldboard plough operated at a fixed tillage depth of 30 cm on a hard, compacted soil in northern Xinjiang (GNSS 46.67524 N, 83.89363 E, 653.9 m; administrative location [TBD]). The usable working length was about 380 m. Acquisition began on 2025-10-27 at 13:33; regional records indicate overcast conditions turning to light rain (about 7-16 C) after two clear days. Gravimetric soil water content was not measured and is treated as a limitation. Seven segments were collected from the same machine on the same field and treated as contiguous portions of one work cycle that can be merged. Fig. 1 shows the GNSS trajectories: repeated parallel passes connected by headland turns at both ends.")
IMG("fig_field_layout.png",w=6.4); CAP("Fig. 1. GNSS trajectories of the seven segments (scale bar 100 m).")
CAP("Table 2. Segment summary (class composition from PC-GMM pseudo-labels)."); TBL(pd.read_csv(os.path.join(RES,"segment_summary.csv")))

H("2.3 Sensors and data acquisition",2)
P("Wheel forces were measured with Kistler RoaDyn S6XT wheel force transducers (WFT), each providing six degrees of freedom (Fx, Fy, Fz and Mx, My, Mz). Force ranges are ±250 kN for Fx and Fz and ±150 kN for Fy; moment ranges are ±50 kNm for Mx and Mz and ±80 kNm for My. Nonlinearity and hysteresis are within 1% of full scale and crosstalk within 1%. Each WFT mounts between the hub and rim and rotates with the wheel; signals are digitized on the wheel and transmitted by near-field telemetry to CAN/Ethernet. Conditioning applies a rotating-to-vehicle coordinate transformation, six-axis decoupling, and temperature, centrifugal and gravity compensation. Two S6XT units were used on the left front and left rear wheels; GNSS provided ground speed; the right-rear wheel speed came from the CAN encoder and the right-front signal was excluded after an encoder fault. All channels shared a hardware-synchronized 200 Hz time base. The WFT acted as a research-grade reference; Section 2.8 shows the same quantities are obtainable on a production vehicle, so the transducer is not a deployment requirement.")

H("2.4 Preprocessing",2)
P("Preprocessing followed a fixed procedure: the leading samples of each segment were removed; impulse spikes in the longitudinal force channels were removed with a Hampel filter; a fourth-order zero-phase Butterworth filter was applied (5 Hz cutoff for force, 8 Hz for generator torque); and GNSS speed was smoothed with a 21-point median filter. The Butterworth magnitude response, applied forward and backward (zero phase), is")
EQ("|H(f)|² = 1 / ( 1 + (f / f_c)^{2 n_b} )",1)
P("and the Hampel rule replaces a sample by the local median m_j when it exceeds three scaled median absolute deviations,")
EQ("m_j = med{ x_{j−w:j+w} } ;   x_j → m_j  if  |x_j − m_j| > 3 · 1.4826 · MAD_j",2)
EQ("x̃_j = med{ x_{j−k:j+k} } ,  k = 10",3)
P("Wheel slip uses GNSS as the ground-speed reference, with low-speed guards and out-of-range clipping; the TCU-estimated speed was retained only for comparison and invalid state-of-charge fill values were set to missing. For wheel i, with rolling radius r_w and angular velocity ω_i,")
EQ("v_w,i = r_w,i · ω_i",4)
EQ("s_i = ( v_w,i − v_g ) / v_w,i",5)
P("where v_g is the GNSS ground speed; slip is undefined at very low speed and set to missing there, then imputed within each fold. The longitudinal draft force is the sum of the two measured left-side wheel forces,")
EQ("F_x = F_x,LF + F_x,LR",6)
P("with forces converted to kilonewtons. Cutoffs were chosen from the bandwidth: a Welch power spectral density of the tillage draft force shows 99.9% of the energy lies below 5 Hz (Fig. S1), so the 5 Hz cutoff retains essentially all tillage dynamics.")

H("2.5 Hierarchical PC-GMM pseudo-labelling",2)
P("Five conditions were defined: idle/stop, headland, transition (implement entry/exit), medium-heavy tillage and heavy tillage. Labels were generated without annotation by a hierarchical, physics-constrained mixture model. A Gaussian mixture density with K components is [15,16]")
EQ("p(x) = Σ_{k=1}^{K} π_k · N(x | μ_k, Σ_k) ,   Σ_k π_k = 1",7)
P("with posterior responsibility")
EQ("γ_k(x) = π_k N(x|μ_k,Σ_k) / Σ_j π_j N(x|μ_j,Σ_j)",8)
P("Parameters are estimated by expectation-maximization; component means are ordered and the physically separated solution with the lowest BIC is retained. The hierarchy is: Level 1, a one-dimensional two-component mixture on draft force separates non-tillage from tillage at the bimodal valley (boundary about 19.9 kN); Level 2a, within non-tillage a two-component mixture on ground speed separates idle from headland; Level 2b, tillage draft force is unimodal, so two free components do not identify natural populations and an ordinal split at the tillage median defines the grades,")
EQ("F* = med{ F_x : tillage } ;   MedHeavy if F_x < F* , else Heavy",9)
P("Transition windows are those mixed between Level-1 components or showing a sharp inter-window load step,")
EQ("Transition = { max_k γ_k < τ } ∪ { |ΔF_x| > δ }",10)
P("with τ = 0.85 and δ = 4 kN. The two tillage grades are thus ordered engineering bins rather than discovered clusters; a complete four-level taxonomy would require variable-depth, variable-soil or variable-gear data. K-means and unconstrained two-dimensional GMM (K=5) on (F_x, v) serve as baselines, and a sensitivity analysis varies τ over 0.80-0.95 and δ over 3-5 kN.")

H("2.6 Feature engineering",2)
P("Features use only the causal, pre-load side. The production-equivalent (CAN-only) set has 19 features spanning vehicle kinematics, wheel-speed variability, front and rear slip, a slip-coverage indicator and speed frequency descriptors; the full (31-feature) set adds draft-force time-domain statistics, force frequency descriptors and the force-speed correlation (Table 3; full dictionary in Table S1). Powertrain power, generator and motor signals, battery power and state of charge never enter either set.")
CAP("Table 3. Feature groups (full dictionary in Table S1)."); TBL(pd.DataFrame([
 ("G_V speed kinematics","v_mean, v_std, v_max, a_std, a_p95","5","CAN"),
 ("G_WHEEL wheel-speed variability","v_LF_std, v_LR_std, v_RR_std","3","CAN"),
 ("G_SLIP_F front slip","Slip_LF_mean, Slip_LF_std","2","CAN"),
 ("G_SLIP_R rear slip","Slip_LR/RR_mean, Slip_LR/RR_std","4","CAN"),
 ("slip coverage","slip_cov","1","CAN"),
 ("G_FREQ_CAN speed frequency","v_lf, v_hf, v_domf, v_ent","4","CAN"),
 ("G_FX draft time domain","Fx_mean/std/cv/range/skew/kurt, dFx_rms","7","Full add-on"),
 ("G_FREQ_FX force frequency","fx_lf, fx_hf, fx_domf, fx_ent","4","Full add-on"),
 ("G_CPL force-speed coupling","rho_fxv","1","Full add-on"),
],columns=["Group","Features","n","Set"]))
P("For a window x(t) of N samples, the time-domain statistics are")
EQ("μ=N⁻¹Σx ;  σ=√(N⁻¹Σ(x−μ)²) ;  CV=σ/μ ;  R=max x−min x ;  γ₁=N⁻¹Σ(x−μ)³/σ³ ;  κ=N⁻¹Σ(x−μ)⁴/σ⁴−3",11)
P("and the load-change rate is the root-mean-square first difference,")
EQ("dFx_rms = √( (1/(N−1)) Σ_{t=2}^{N} (F_x,t − F_x,t−1)² )",12)
P("Frequency features use a Welch periodogram (6-s segments, 50% overlap),")
EQ("P(f) = (1/(M·U)) Σ_{m=1}^{M} | Σ_{n=0}^{L−1} x_m[n] w[n] e^{−j2π f n} |² ,  U = L⁻¹ Σ w²",13)
EQ("R_LF = Σ_{f<f_b} P(f) / Σ_f P(f) ,   R_HF = 1 − R_LF",14)
P("where R_LF/R_HF are the low/high band powers and the dominant frequency f_dom is the spectral peak. Spectral entropy is")
EQ("H = − Σ_j p_j ln p_j / ln N_b",15)
P("and the force-speed coupling is the Pearson correlation,")
EQ("ρ_fxv = Σ(Fx−μ_F)(v−μ_v) / ( √Σ(Fx−μ_F)² · √Σ(v−μ_v)² )",16)

H("2.7 Recognition models and training",2)
P("Six classifiers spanning ensemble, kernel, neural and sequence families were compared: random forest [10], a radial-basis support vector machine [11], gradient-boosted trees (XGBoost) [9], a multilayer perceptron, a two-layer long short-term memory network [12] and a temporal convolutional network [13] with dilated residual blocks. Configurations are listed in Table 4.")
CAP("Table 4. Model configurations."); TBL(pd.DataFrame([
 ("RF","400 trees, Gini, balanced class weights"),
 ("SVM","RBF kernel, C=10, gamma=scale, balanced weights"),
 ("XGBoost","400 estimators, depth 4, eta=0.05, softmax objective"),
 ("MLP","hidden 128-64, ReLU, L2 regularization, Adam"),
 ("LSTM","2 layers, 64 hidden, dropout 0.2; series 60 pts"),
 ("TCN","3 residual blocks, 32 channels, kernel 3, dilations 1/2/4"),
],columns=["Model","Configuration"]))
P("Neural and sequence models were trained with Adam (learning rate 1e-3, weight decay 1e-4), batch size 64, for 35 epochs, using cross-entropy with inverse-frequency class weights; channels were standardized within each fold as z = (x − μ_fold)/σ_fold. The 60-point series was obtained by resampling the 3-s window to 20 Hz. Sequence inputs carry draft force, speed and three-wheel slip; production variants omit draft force.")

H("2.8 Validation protocol and deployment",2)
P("Because the seven segments are independent acquisition starts on the same field, a leave-one-segment-out (LOSO) scheme with seven folds was used, each holding out one segment so performance reflects generalization to an unseen pass. Imputation, standardization and class weights were estimated strictly within each fold. Aggregating out-of-fold predictions gives the confusion matrix, accuracy, macro-F1, per-class F1 and balanced accuracy; fold-level macro-F1 is reported as mean ± standard deviation. Pairwise significance used McNemar's test on out-of-fold predictions and paired fold-level tests; because overlapping windows are not fully independent, the fold-level tests are the conservative reference.")
P("For deployment, an engineering subset of ten production-equivalent features was selected by XGBoost gain. A compact multilayer perceptron (64-32 units) was post-training quantized to INT8; static min-max observers were calibrated on the training fold and weights and activations mapped by")
EQ("x_q = clamp( round(x / s) + z , q_min , q_max )",17)
P("where s and z are the scale and zero point. FP32 and INT8 models were compared on LOSO accuracy, macro-F1, size and single-window latency measured single-threaded on the CPU; latency is an x86 proxy for an embedded target without a fixed part number. On a production tractor the draft load can alternatively be estimated from driveline torque or a longitudinal model,")
EQ("F_d ≈ T_m · i_g · η_t / r",18)
P("and slip is computed from wheel and GNSS speed.")

# ---------------- 3 Results ----------------
H("3. Results")
H("3.1 Pseudo-label validation",2)
P("The PC-GMM produced 950 idle, 464 headland, 128 transition, 816 medium-heavy and 842 heavy windows. Mean draft force was strictly ordered (5.4, 6.6, 22.0, 32.5 and 36.9 kN) and the pseudo-labels agreed with the independent physics reference at 97%. Table 5 shows PC-GMM reached ARI 0.921 and 97% aligned accuracy versus 0.43/61% for K-means and 0.59-0.63/69-80% for unconstrained GMM, which grouped tillage by speed rather than load and did not separate the two grades.")
CAP("Table 5. Clustering baseline comparison vs the physics reference."); TBL(pd.read_csv(os.path.join(RES,"clustering_baseline.csv")))
IMG("fig1_flowchart.png"); CAP("Fig. 2. Overall framework; powertrain power is used only for posterior validation.")
H("3.2 Model comparison",2)
P("Table 6 and Fig. 3 summarize LOSO performance. With the full set, random forest and XGBoost reached 0.973 accuracy and 0.920 macro-F1; the SVM reached 0.909/0.871, the MLP 0.893/0.857 and the sequence models 0.911-0.922 accuracy and 0.847-0.862 macro-F1. On production signals XGBoost was best at 0.842/0.764. Statistical testing supported the ranking: on the full set XGBoost and RF did not differ (p=0.89) but both significantly outperformed the SVM, MLP, LSTM and TCN (McNemar p<0.001; fold-level Wilcoxon p=0.016-0.03); on production signals XGBoost and RF were tied (p=0.66), with weaker fold-level evidence over the rest. Multi-seed runs (Table S2) confirm the deep-model results are stable (standard deviation ≤0.012) and remain below the tree ensembles.")
IMG("fig_model_compare.png"); CAP("Fig. 3. Macro-F1 of six classifiers for the production (19) and full (31) sets under LOSO.")
CAP("Table 6. LOSO performance of six classifiers."); TBL(pd.DataFrame([
 ("RF",.840,.741,.973,.920),("SVM",.816,.755,.909,.871),("XGBoost",.842,.764,.973,.920),
 ("MLP",.816,.727,.893,.857),("LSTM",.817,.739,.911,.847),("TCN",.817,.753,.922,.862)],
 columns=["Model","Acc CAN","F1 CAN","Acc Full","F1 Full"]))
H("3.3 Confusion analysis",2)
P("Fig. 4 shows the aggregated LOSO confusion matrices. With the full set, idle, headland, medium-heavy and heavy were recalled at 99%, 97%, 98% and 99%; transition was recalled at 65%, with residual samples assigned to adjacent conditions. On production signals idle and headland stayed high (98%, 91%), transition fell to 38%, and the two tillage grades were recalled at 76% and 79%, with most errors between the grades that are defined by draft force.")
IMG("fig_confusion.png"); CAP("Fig. 4. Aggregated LOSO confusion matrices for XGBoost: (a) production, (b) full set.")
H("3.4 Ablation and wheel slip",2)
P("Progressive ablation (macro-F1) gave 0.704 for speed kinematics, 0.730 after wheel-speed variability, 0.749 after front slip, 0.767 after rear slip, 0.764 after speed frequency features, and 0.921 once draft-force time-domain statistics were added; force frequency features and the force-speed correlation did not change the result. A single wheel contributed similarly (front 0.749, left-rear 0.753, right-rear 0.751) and three-wheel fusion was best (0.769). Draft force is therefore decisive, slip gives a small gain maximized by multi-wheel fusion, and frequency features add no measurable advantage.")
H("3.5 Principal-component analysis",2)
P("The first three components explained 28.5%, 15.6% and 11% of the variance (72% over six). PC1 is a dynamic-intensity axis, PC2 a speed-load axis and PC3 a slip-adhesion axis dominated by wheel slip, explaining why slip retains a small gain beyond speed and force.")
IMG("fig_pca.png"); CAP("Fig. 5. PCA: (a) scree, (b) PC1-PC2 by condition, (c) loadings; slip loads on PC3.")

# ---------------- 4 Discussion ----------------
H("4. Discussion")
H("4.1 Physical structure of the merged cycle",2)
P("The merged work cycle is strongly bimodal in draft force: the tractor is either manoeuvring at low force or ploughing near full load, with almost no stable samples between 15 and 27 kN. This gap is the expected consequence of a fixed-depth pass on one field and explains why an intermediate half-load condition cannot be trained here; the two tillage grades are best reported as ordered bins of a unimodal distribution rather than natural clusters.")
H("4.2 Why the models and features behave as they do",2)
P("Tree ensembles matching or exceeding neural and sequence models is consistent with the data regime: the sample is moderate, the tabular features already summarize the within-window dynamics, and the decision boundary between ordered load grades is low-dimensional and axis-aligned, which boosted trees partition efficiently, whereas neural nets have more capacity than the task rewards and are estimated less precisely from seven folds. The ablation and PCA together give a mechanistic account. Draft force is decisive because it directly measures the implement load; adding it closes most of the error. Wheel slip carries a small independent signal because it reflects tyre-terrain adhesion that is not identical to force or speed, and it loads on a separate PC3; fusing three wheels averages out single-wheel noise and is best, while any one wheel is similar. Frequency features add nothing because the tillage signal is almost entirely below 5 Hz and its level is already captured by the time-domain amplitude and variability, making band powers and spectral entropy redundant; this is why frequency content is reported as supporting material rather than a core contribution. The transition class is intrinsically the hardest: it occupies only 4% of the cycle, lasts a few windows, and by definition spans two conditions, so even the full-feature model recalls about two thirds of it; for an EMS this residual can be handled conservatively rather than by forcing a precise label.")
H("4.3 Posterior powertrain validation",2)
P("Using the labels as fixed, motor and generator power rose from idle (~0 kW) through headland (~5 kW) and transition (~39 kW) to tillage (~140 kW). The heavy grade had a slightly lower whole-grade average motor power because it ran at lower speed (1.41 vs 1.69 m/s) against a larger force; when speed is matched in narrow bins, the heavy grade always shows higher draft force and motor power, confirming the labels are valid and the univariate comparison is confounded by speed. The series-hybrid balance between motor demand and generator-plus-battery supply,")
EQ("P_mot ≈ |P_gen| + P_bat",19)
P("held at r = 0.996 (median supply/demand 1.09).")
IMG("fig_posthoc_power.png"); CAP("Fig. 6. Posterior validation: power by condition, Fx-v hyperbolae, matched-speed comparison and power balance.")
H("4.4 Engineering deployability and sensor equivalence",2)
P("The signals are not specific to an instrumented prototype. The WFT acted as a research-grade reference for wheel forces and angular velocities; on a production tractor the same quantities come from the standard CAN bus (wheel encoders, motor/generator torque) and a low-cost GNSS receiver, with slip from wheel and ground speed and draft load from Eq. (18). The transducer is a calibration benchmark, not a deployment requirement. CAN-versus-GNSS speeds correlated at 0.97; the small offset reflects wheel slip and is correctable, while the encoder-based rear slip had larger variance, indicating that production filtering, multi-wheel fusion and calibration govern on-vehicle accuracy.")
CAP("Table 7. Edge deployment on the ten-feature engineering subset."); TBL(pd.read_csv(os.path.join(RES,"deployment_int8.csv")))
P("The quantized network retained macro-F1 0.757 in an 11.2 kB model with ~81 microsecond per-window inference versus the 2.4 MB, ~347 microsecond tree model; the full-feature INT8 model retained macro-F1 0.860. With a 1.5 s window step, inference occupies a negligible fraction of the cycle, supporting real-time embedded operation.")
H("4.5 Sensitivity and limitations",2)
P("The pseudo-labels are robust: δ = 4 kN gives the highest agreement (0.97), δ = 3 over-flags and δ = 5 under-flags transitions, while τ over 0.80-0.95 has little effect because the Level-1 valley is sharp and transitions here are detected mainly by the load step. Limitations are that all segments come from one machine, field, depth and soil; an intermediate load level is absent and the four-level taxonomy needs variable depth, soil, moisture and gear data; soil water content was not measured; encoder-based slip was noisier than the WFT reference; and the closed-loop effect of the recognizer within a full EMS is future work.")

# ---------------- 5 Conclusions ----------------
H("5. Conclusions")
P("A physics-constrained, annotation-free operating-condition recognizer was demonstrated for a series-hybrid tractor on merged field data. The hierarchical PC-GMM generated five-condition pseudo-labels at 97% agreement with an independent reference without using powertrain power, versus 61-80% for unconstrained clustering, and tree ensembles reached 0.973 accuracy and 0.920 macro-F1 with research-grade draft force and 0.842/0.764 on production-equivalent signals. Draft force was decisive, wheel slip added a small gain largest with three-wheel fusion, and frequency features added none; matched-speed checks and a 0.996 power-balance correlation confirmed physical validity. An INT8 network retained production accuracy in an 11 kB, 81-microsecond model. Hybrid-tractor condition recognition can therefore be built without field annotation or special-vehicle hardware and deployed on low-cost embedded controllers as a practical front end for predictive energy management.")

# ---------------- Supplementary ----------------
H("Supplementary material",1)
P("Fig. S1: Welch PSD of draft force and speed (99.9% of draft energy below 5 Hz). Fig. S2: clustering baseline scatter. Fig. S3: PC-GMM sensitivity. Table S1: full feature dictionary. Table S2: deep-model multi-seed results.",j=False)
CAP("Table S2. Deep-model macro-F1 over three seeds (LOSO)."); TBL(pd.read_csv(os.path.join(RES,"deep_multiseed.csv")))

# ---------------- Declarations ----------------
H("Declarations")
P("Funding: supported by [TBD: Henan Provincial Department of Science and Technology Major Project, grant TBD]. Author contributions: [TBD]. Conflicts of interest: [TBD]. Data availability: [TBD]. Acknowledgements: [TBD].",j=False)

# ---------------- References ----------------
H("References")
refs=[
 "Tellioğlu, A.C., Yüce, H., Kesen, U., Dana, A., 2026. Predicting equipment utilization in agricultural tractors using field data and machine learning. Journal of Mechatronics and Artificial Intelligence in Engineering. https://www.extrica.com/article/26673",
 "An, Q., Wang, F., Liu, Y., Deng, X., Zhou, L., Zhao, B., Wei, L., 2025. Lightweight RepViT-based working condition recognition method for agricultural implements. Transactions of the Chinese Society for Agricultural Machinery 56 (2).",
 "Zhai, W.X., Lyu, Z., Li, Z.C., Pan, J.W., Li, D.F., Sun, D.M., et al., 2026. SSL-GAT: a self-supervised learning-based graph attention network for agricultural machinery trajectory operation mode identification. Int. J. Agric. Biol. Eng. 19 (2), 282-293. https://doi.org/10.25165/j.ijabe.20261902.9966",
 "Pan, C.Y., Zhai, W.X., 2025. AMGFormer: a multilevel GNSS recording autoencoder for agricultural machinery trajectory operation mode identification. Computers and Electronics in Agriculture 237, 110638.",
 "Abbaspour-Gilandeh, Y., Fazeli, M., Roshanianfard, A., Hernández-Hernández, M., Gallardo-Bernal, I., Hernández-Hernández, J.L., 2020. Prediction of draft force of a chisel cultivator using artificial neural networks and its comparison with regression model. Agronomy 10 (4), 451. https://doi.org/10.3390/agronomy10040451",
 "Feng, G., Zhang, J., Yan, X., Dong, C., Liu, M., Xu, L., 2024. Research on energy-saving control of agricultural hybrid tractors integrating working condition prediction. PLoS One 19 (3), e0299658. https://doi.org/10.1371/journal.pone.0299658",
 "Deng, X., 2026. Slip-ratio-aware energy management of a hybrid tractor under variable plowing loads using a DP-calibrated ECMS. Agriculture 16 (18), 2000. https://doi.org/10.3390/agriculture16182000",
 "Kaufman, S., Rosset, S., Perlich, C., Stitelman, O., 2012. Leakage in data mining: formulation, detection, and avoidance. ACM Transactions on Knowledge Discovery from Data 6 (4), 1-21. https://doi.org/10.1145/2382577.2382579",
 "Chen, T., Guestrin, C., 2016. XGBoost: a scalable tree boosting system. Proc. 22nd ACM SIGKDD, 785-794. https://doi.org/10.1145/2939672.2939785",
 "Breiman, L., 2001. Random forests. Machine Learning 45 (1), 5-32. https://doi.org/10.1023/A:1010933404324",
 "Cortes, C., Vapnik, V., 1995. Support-vector networks. Machine Learning 20 (3), 273-297.",
 "Hochreiter, S., Schmidhuber, J., 1997. Long short-term memory. Neural Computation 9 (8), 1735-1780.",
 "Bai, S., Kolter, J.Z., Koltun, V., 2018. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv:1803.01271.",
 "Dempster, A.P., Laird, N.M., Rubin, D.B., 1977. Maximum likelihood from incomplete data via the EM algorithm. J. R. Stat. Soc. B 39 (1), 1-38.",
 "Bishop, C.M., 2006. Pattern Recognition and Machine Learning. Springer.",
 "Wong, J.Y., 2008. Theory of Ground Vehicles, 4th ed. Wiley.",
 "Brixius, W.W., 1987. Traction prediction equations for 4WD tractors. ASAE Paper 87-1622.",
 "Liakos, K.G., Busato, P., Moshou, D., Pearson, S., Bochtis, D., 2018. Machine learning in agriculture: a review. Sensors 18 (8), 2674. https://doi.org/10.3390/s18082674",
 "Molari, G., Mattetti, M., Lenzini, N., Fiorati, S., 2019. An updated methodology to analyse the idling of agricultural tractors. Biosystems Engineering 187, 160-170.",
 "Lee, J.W., Kim, J.S., Kim, K.U., 2016. Computer simulations to maximise fuel efficiency and work performance of agricultural tractors in rotovating and ploughing operations. Biosystems Engineering 142, 1-11.",
 "ASABE, 2020. Agricultural Machinery Management Data, ASABE D497.",
]
for i,rf in enumerate(refs,1):
    p=doc.add_paragraph(f"[{i}] {rf}"); p.paragraph_format.line_spacing=1.15
    for r in p.runs: r.font.size=Pt(10)

out=os.path.join(PROJ,"manuscript","HB2204_manuscript_expanded.docx")
try:
    doc.save(out)
except PermissionError:
    out=os.path.join(PROJ,"manuscript","HB2204_manuscript_v3.docx"); doc.save(out)
print("saved",out)
