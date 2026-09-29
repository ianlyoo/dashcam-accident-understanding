# Official submissions

Source: the project's local `tracking/submissions.csv` ledger (frozen after the competition). Only attempts counted by the daily submission limit appear here; local-only candidates are discussed in [the journey](journey.md). Scores are public leaderboard scores. **Kept** means the row set a new best total at that point, based on ledger order; the final selected row was S182_full.

| ID | Date (KST) | Change | Total | S1 | S2 | S3 | Status | Kept |
|---|---|---|---:|---:|---:|---:|---|---|
| S001 | 2026-08-29 | Official starter baseline sentinel | 0.2062892669 | 0.4045598914 | 0.1647156642 | 0.1487275574 | success | yes |
| S002 | 2026-08-29 | Replace Stage 1 only with deterministic spectral forensic threshold | 0.2465537592 | 0.6058823529 | 0.1647156642 | 0.1487275574 | success | yes |
| S003 | 2026-08-29 | Replace Stage 3 only with deterministic optical-flow ego-motion | NA | NA | NA | NA | failed | no |
| S004 | 2026-08-30 | Stage 3 ego-motion identical to S003 with dataclass-free parameter containers | 0.3411082328 | 0.4045598914 | 0.1647156642 | 0.4857749721 | success | yes |
| S005 | 2026-08-30 | Combine S002 Stage 1 champion and S004 Stage 3 champion | 0.3813727251 | 0.6058823529 | 0.1647156642 | 0.4857749721 | success | yes |
| S006 | 2026-08-30 | S005 champion plus collision_frame-only deterministic Stage 2 motion override | 0.4202048419 | 0.6058823529 | 0.2617959562 | 0.4857749721 | success | yes |
| S007 | 2026-08-31 | S006 champion plus fixed nine-frame lead entry_frame | 0.4263362287 | 0.6058823529 | 0.2771244233 | 0.4857749721 | success | yes |
| S008 | 2026-09-01 | S007 champion plus COCO vehicle-track entry_side | 0.4357457099 | 0.6058823529 | 0.3006481262 | 0.4857749721 | success | yes |
| S009 | 2026-09-01 | S008 plus three-feature Stage 1 ensemble and 720-pixel canonicalization | 0.3889033324 | 0.3716704656 | 0.3006481262 | 0.4857749721 | success | no |
| S010 | 2026-09-01 | S008 champion plus cached independent-motion entry onset | 0.4439208924 | 0.6058823529 | 0.3210860824 | 0.4857749721 | success | yes |
| S012 | 2026-09-02 | S010 plus cached target-aware lateral-clearance evasion_space | 0.4505041336 | 0.6058823529 | 0.3375441855 | 0.4857749721 | success | yes |
| S013 | 2026-09-02 | S010 plus native-resolution three-feature Stage 1 ensemble | 0.3907247174 | 0.3399014778 | 0.3210860824 | 0.4857749721 | success | no |
| S014 | 2026-09-02 | S012 champion plus comma2k19-supervised hybrid Stage 3 steering head | 0.4739199221 | 0.6058823529 | 0.3375441855 | 0.5443144432 | success | yes |
| S015 | 2026-09-03 | S014 plus five-seed temporal steering decoder | 0.4838155394 | 0.6058823529 | 0.3375441855 | 0.5690534866 | success | yes |
| S019 | 2026-09-03 | S015 plus supervised temporal acceleration head | 0.4969933892 | 0.6058823529 | 0.3375441855 | 0.6019981111 | success | yes |
| S021 | 2026-09-03 | S019 supervised acceleration plus S018 dense temporal steering | 0.4839967655 | 0.6058823529 | 0.3375441855 | 0.5695065518 | success | no |
| S024 | 2026-09-04 | S019 and S022 acceleration probability blend | 0.4993989957 | 0.6058823529 | 0.3375441855 | 0.6080121274 | success | yes |
| S027 | 2026-09-04 | S019 plus per-file capture-chain classifier | 0.4625550139 | 0.4336904762 | 0.3375441855 | 0.6019981111 | success | no |
| S036 | 2026-09-04 | S019 plus target-anchored Scoop poly2 Stage 1 with S002 ambiguity fallback | 0.5008062706 | 0.6249467600 | 0.3375441855 | 0.6019981111 | success | yes |
| S038 | 2026-09-05 | S036 Stage 1 plus S024 Stage 3 with unchanged Stage 2 | 0.50321187716 | 0.62494676 | 0.3375441855 | 0.6080121274 | success | yes |
| S044 | 2026-09-05 | S036 Stage 1 plus pure S022 robust acceleration with unchanged Stage 2 and steering | 0.50158265608 | 0.62494676 | 0.3375441855 | 0.6039390747 | success | no |
| S048 | 2026-09-05 | S038 with mixed-cadence ordered temporal acceleration context | 0.51548930768 | 0.62494676 | 0.3375441855 | 0.6387057037 | success | yes |
| S050 | 2026-09-06 | S048 mixed20and10Hz X14 steering weights only | 0.50964733812 | 0.62494676 | 0.3375441855 | 0.6241007798 | success | no |
| S051 | 2026-09-06 | S048 temporal360 acceleration robust alpha .50 to .75 | 0.5169099506 | 0.62494676 | 0.3375441855 | 0.642257311 | success | yes |
| S052 | 2026-09-06 | S048 temporal360 acceleration robust alpha .50 to .25 | 0.5139788924 | 0.62494676 | 0.3375441855 | 0.6349296655 | success | no |
| S053 | 2026-09-07 | S051 temporal360 robust alpha .75 to1.00; all weights unchanged | 0.51694868968 | 0.62494676 | 0.3375441855 | 0.6423541587 | success | yes |
| S054 | 2026-09-07 | S053 Stage2 detector replacement SSDLite320 to FasterRCNN ResNet50 FPNv2; otherlogicfixed | 0.51340246496 | 0.62494676 | 0.3286786237 | 0.6423541587 | success | no |
| S055 | 2026-09-07 | S053 with anchored nativeRGB residualCNN Stage1 only; Stage2and3unchanged | 0.49723296952 | 0.5263681592 | 0.3375441855 | 0.6423541587 | success | no |
| S059 | 2026-09-08 | OnlyS053steeringNPZ replaced bymirror-trainedS015ensemble | 0.51273173956 | 0.62494676 | 0.3375441855 | 0.6318117834 | success | no |
| S060 | 2026-09-08 | SpatialRAFT grid andtemporalaccelerationhead onS053; quarterobservations shareddecode; steeringandStage1/2preserved | 0.51252089124 | 0.62494676 | 0.3375441855 | 0.6312846626 | success | no |
| S062 | 2026-09-08 | S053 collision tail eligibility only; complete incumbent call preserved | 0.51694868968 | 0.62494676 | 0.3375441855 | 0.6423541587 | success | no |
| S069 | 2026-09-09 | S053 complete Stage2 call preserved then entry_side replay in full-input sequence coordinates | 0.51694868968 | 0.62494676 | 0.3375441855 | 0.6423541587 | success | no |
| S075 | 2026-09-10 | Full306 canonical720 MLP withunchangedS053Stage2/3 | NA | NA | NA | NA | error | no |
| S076 | 2026-09-10 | Full306 canonical720 C=.1L2 linear withTRAIN-onlyweightedstatistics | NA | NA | NA | NA | error | no |
| S077 | 2026-09-10 | Full306 sharedMLP trained200native200canonicalupdates; one-decode dualmeaninference | NA | NA | NA | NA | error | no |
| S078 | 2026-09-11 | S076 compatibility repair; optional position metadata withunchangedordinalsamplingandweights | 0.52999218288 | 0.690164226 | 0.3375441855 | 0.6423541587 | success | yes |
| S079 | 2026-09-11 | Existingfull306S077dualMLP withexactscoredS078decoder; mainandStage2/3matchS078 | 0.55837537778 | 0.8320802005 | 0.3375441855 | 0.6423541587 | success | yes |
| S087 | 2026-09-11 | User-authorizedresumptionoforiginalfull24S084physicalDINOrecipe | 0.55370641162 | 0.8087353697 | 0.3375441855 | 0.6423541587 | success | no |
| S088 | 2026-09-12 | S079sameinferencewithofficialTRAINclassificationmass.50andScoopDLC.25each | 0.55555247744 | 0.8179656988 | 0.3375441855 | 0.6423541587 | success | no |
| S089 | 2026-09-12 | S079withrobustStage3accelerationheadtrainedonRAV188plusCivic202;allinferencesourceandotherheadsunchanged | 0.55703931250 | 0.8320802005 | 0.3375441855 | 0.6390139955 | success | no |
| S090 | 2026-09-12 | FrozenDINOv2Base6144replacesSmall3072withsameS079306FITrecipe;Stage2/3unchanged | 0.55730481472 | 0.8267273852 | 0.3375441855 | 0.6423541587 | success | no |
| S091 | 2026-09-13 | ExactpreviouslyunsubmittedS083halfS078linearhalfS079dualMLPwithsame72Smallviews | 0.54569926864 | 0.7686996548 | 0.3375441855 | 0.6423541587 | success | no |
| S093 | 2026-09-13 | SameS092datawithmatchingS079foldinitializationand100updateslr3e-5freshAdamW | 0.55821391644 | 0.8312728938 | 0.3375441855 | 0.6423541587 | success | no |
| S094 | 2026-09-13 | FixedequalprobabilityS090BaseandS093adaptedSmallwithoneshareddecode144views;Stage2/3unchanged | 0.56301929790 | 0.8552998011 | 0.3375441855 | 0.6423541587 | success | yes |
| S095 | 2026-09-14 | S094Smallhead100stepadaptationwithold492FITand64positiveVDemoireJPEGseqplusteacherretention;Baseand144viewsunchanged | 0.54796309002 | 0.7800187617 | 0.3375441855 | 0.6423541587 | success | no |
| S096 | 2026-09-14 | RestoreexactS079SmallheadinS094fixedhalfS090Baseensemble;zerofitsandunchanged144views | 0.56089126672 | 0.8446596452 | 0.3375441855 | 0.6423541587 | success | no |
| S097 | 2026-09-14 | S094BaseheadwarmadaptationfrommatchingS090weights100stepslr3e-5onexisting492FIT;Smalland144viewsunchanged | 0.56527484342 | 0.8665775287 | 0.3375441855 | 0.6423541587 | success | yes |
| S098 | 2026-09-15 | ExactS097headswithfixed.75Base+.25Smallfinalprobability;oneexpressionandnoticeonly | 0.56313723998 | 0.8558895115 | 0.3375441855 | 0.6423541587 | success | no |
| S100 | 2026-09-15 | ExactS097pluscanonical-onlyfrozenLarge8192branchandsixscratchheads;fixedhalfS097halfLarge;180views | 0.56228933556 | 0.8516499894 | 0.3375441855 | 0.6423541587 | success | no |
| S101 | 2026-09-15 | ExactS097runtimewithtwoheadsadapted100updatesusingbalancedJPEG90andcleanlogitretention;144views | 0.56685395126 | 0.8744730679 | 0.3375441855 | 0.6423541587 | success | yes |
| S102 | 2026-09-16 | Additional100-stepJPEGdeliverycontinuationfromS101withoriginalS097teacherandfreshoptimizer;unchanged144-viewinference | 0.56685395126 | 0.8744730679 | 0.3375441855 | 0.6423541587 | success | no |
| S103 | 2026-09-16 | ExportalreadysealedS102cleancontrolsecond100-stepblockfromS101withS097teacher;nonewfits | 0.56606561788 | 0.870531401 | 0.3375441855 | 0.6423541587 | success | no |
| S104 | 2026-09-16 | TwoStage1headscontinued100updatesonfull-streamH264CRF23exposurefromS101withoriginalS097teacher;backbonesdecoderStage2/3equalmixtureand144views... | 0.56375001518 | 0.8589533875 | 0.3375441855 | 0.6423541587 | success | no |
| S105 | 2026-09-17 | ExportsealedS101first-blockcleancontrolheadsoverthescoredS101ZIP;fivereplacementsplusonenotice;41memberscopiedbyte-identical;nonewfits | 0.56763991844 | 0.8784029038 | 0.3375441855 | 0.6423541587 | success | yes |
| S106 | 2026-09-17 | Exportthedeliveredcontrolsecond-blockcontinuationoverthescoredS105ZIP;48members;fivereplacementsplusonenotice;samegraph;fourmatchedfits | 0.56527484342 | 0.8665775287 | 0.3375441855 | 0.6423541587 | success | no |
| S108 | 2026-09-17 | ExactS105plusfrozenVJEPA2.1encoderandoneF4CANsupervisedaccelhead;Stage1Stage2andsteeringpreserved | 0.52850613188 | 0.8784029038 | 0.3375441855 | 0.5445196923 | success | no |
| S109 | 2026-09-18 | 75percentS105robustmotionplus25percentS108videoprobabilitiesthenexactS105acceldecoder;nochangedweightsornewfits | 0.57141462928 | 0.8784029038 | 0.3375441855 | 0.6517909358 | success | yes |
| S110 | 2026-09-18 | Video probability coefficient25to50percent over exactS109; allweights decoder sampling Stage1 Stage2 steering unchanged | 0.56161482060 | 0.8784029038 | 0.3375441855 | 0.6272914141 | success | no |
| S111 |  | Video coefficient25to12.5percent over exactS109; allweights decoder sampling Stage1 Stage2 steering unchanged | 0.56950817452 | 0.8784029038 | 0.3375441855 | 0.6470247989 | success | no |
| S112 | 2026-09-19 | Video coefficient25to31.25percent over exactS109; allweights decoder sampling Stage1 Stage2 steering unchanged | 0.57051833176 | 0.8784029038 | 0.3375441855 | 0.649550192 | success | no |
| S113 | 2026-09-19 | Video coefficient25to21.875percent over exactS109; allweights decoder sampling Stage1 Stage2 steering unchanged | 0.57125605436 | 0.8784029038 | 0.3375441855 | 0.6513944985 | success | no |
| S114 | 2026-09-19 | NewSAMA-supervisedResNet18FPNlanegeometryandcollisionanchoredactorentryoverexactS109;entryonly | 0.56937083368 | 0.8784029038 | 0.3324346965 | 0.6517909358 | success | no |
| S115 | 2026-09-20 | ExactS109plusmulti-actorcollisioncomparison;collision_frameonly;nonewweights | 0.57141462928 | 0.8784029038 | 0.3375441855 | 0.6517909358 | success | no |
| S116 | 2026-09-20 | ExactS109plusstrictlane-relativeentryoverride;entry_frameonly;reusedS114weights | 0.57141462928 | 0.8784029038 | 0.3375441855 | 0.6517909358 | success | no |
| S118 | 2026-09-24 | S109 plus Stage2 collision floor_replace (S109 position <30 -> 32) and Stage3 steer constant STRAIGHT (diagnostic); ZIP 25c352f0499edcba | 0.52765500608 | 0.8784029038 | 0.3375441855 | 0.5423918778 | success | no |
| S120 | 2026-09-24 | S109 plus learned CCD collision localiser (DINOv2-Base + 5-seed conv head) collision only; ZIP c671ed6f3878b4c5 | 0.57141462928 | 0.8784029038 | 0.3375441855 | 0.6517909358 | success | no |
| S121 | 2026-09-24 | S120 plus entry = learned collision - 5 frames; ZIP 56bbd89b29bc95e3 | 0.57141462928 | 0.8784029038 | 0.3375441855 | 0.6517909358 | success | no |
| S122 | 2026-09-25 | S109 plus learned CCD collision localiser with 10 Hz grid resampling for 50<n<=310 frames (S120 fix); collision only; ZIP ca470c5c45fd8c10; ro... | 0.57039273148 | 0.8784029038 | 0.334989441 | 0.6517909358 | success | no |
| S124 | 2026-09-25 | S109 plus Stage3 steering = 75% S015 raw + 25% V-JEPA steer head on the shared S108 cells; accel and Stage1/2 exact S109; ZIP 66ef2f4455d63cf7... | 0.57726268920 | 0.8784029038 | 0.3375441855 | 0.6664110856 | success | yes |
| S129 | 2026-09-25 | S109 Stage2 collision peak searched only in clip positions [0.42n,0.58n] for n>310 (entry/side/evasion re-anchor through S109) plus Stage3 ste... | 0.55240097052 | 0.8784029038 | 0.2797514245 | 0.6620495499 | success | no |
| S136 | 2026-09-26 | S1 encoder-signature rule (FMP4 + Lavf58.12.100 -> ORIGINAL) + S2 S132 tracker entry/side on S109 collision + S3 S131 accel relabel +-0.25 wit... | 0.45448363688 | 0.3463414634 | 0.3315224457 | 0.6315159148 | success | no |
| S137d | 2026-09-26 | Exact S124 with only Stage2 collision_frame from the S126/S135 Nexar collision re-ranker (top-10 S109 motion maxima re-ranked by GBDT; single... | 0.59463495196 | 0.8784029038 | 0.3809748424 | 0.6664110856 | success | yes |
| S142s | 2026-09-26 | Exact S124 S1 + Stage2 collision_frame from S142 box-feature Nexar re-ranker (S135 motion candidates + S132 detector box/contact features; sin... | 0.6256282022 | 0.8784029038 | 0.4141865213 | 0.7106825323 | success | yes |
| S145_ef | 2026-09-27 | Exact S142s with only Stage2 entry replaced by the S130_ef first-frame entry rule (collision/side/evasion/S1/S3 identical); ZIP 8393cfd3d3f430... | 0.60110265476 | 0.8784029038 | 0.3528726527 | 0.7106825323 | success | no |
| S144s_rr3 | 2026-09-27 | Exact S142s with only Stage2 collision_frame from the S144 E10_base re-ranker ensemble (DINOv2-embed ctx_contact + top-20 ctx_contact + S142-e... | 0.63278148688 | 0.8784029038 | 0.432069733 | 0.7106825323 | success | yes |
| S148_acc | 2026-09-27 | Exact S142s with only Stage3 accel replaced by a HistGBT residual correction on S124 accel probabilities + S141 signals + motion features (ste... | 0.5910303462 | 0.8784029038 | 0.4141865213 | 0.6241878923 | success | no |
| S170_all | 2026-09-28 | Exact S156_casc + S164 runtime cut + S163 one-way S1 recapture cue + S160 collision temporal localizer (2 HGB + 10 nets; entry clamp min(S109... | 0.63792907048 | 0.8784029038 | 0.4448434556 | 0.7107777687 | success | yes |
| S171_ent | 2026-09-28 | Exact S170_all + S161 entry-crossing arm (entry = S161 crossing anchored on S160 collision, then clamp to collision; errors keep S170 entry);... | 0.65632323104 | 0.8784029038 | 0.490828857 | 0.7107777687 | success | yes |
| S172_vis | 2026-09-28 | Exact S171_ent + DINOv2-L dense visual collision localizer (2.5 Hz fp16 batch 16 coarse pass + fusion with S160 candidates; 600 s / 45 min gua... | 0.66245461792 | 0.8784029038 | 0.5061573242 | 0.7107777687 | success | yes |
| S185_nob | 2026-09-29 | Exact S172_vis + S174 entry coverage + S178 side on accepted S161 crossings + S180 +0.2 s on S174-only crossings + S181 collision-gap fallback... | 0.68547919048 | 0.8784029038 | 0.5637187556 | 0.7107777687 | success | yes |
| S182_full | 2026-09-29 | S185 + S176 bucket entry rules (inside_from_start -> first frame; inside_when_first_seen -> first obs - 0.2 s); ZIP 9307c6001253a07e; row 106007 | 0.68740334776 | 0.8784029038 | 0.5685291488 | 0.7107777687 | success | yes |
| S175_stk | 2026-09-29 | Exact S172_vis + S174 entry coverage only (no S176/S178/S180/S181); ZIP 3827273f061a7222; row 106087 | 0.67165169820 | 0.8784029038 | 0.5291500249 | 0.7107777687 | success | no |

An error or failed attempt may have no score. A later candidate can beat an earlier row without becoming the final best. The final public leaderboard rank is reported in the [README](../README.md).
