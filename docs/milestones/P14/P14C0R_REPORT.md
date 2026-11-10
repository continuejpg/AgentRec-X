# P14-C0-R 鈥?Product Type/Role 鏈€灏忎慨澶嶆姤鍛?
> 鑼冨洿锛氫粎 P14-C0-R銆備慨澶嶁€滈敊璇晢鍝佺被鍨?瑙掕壊琚?FINALIZE鈥濈殑闂锛涗笉鏀规帹鑽愭潈閲嶃€佷笉璁粌/SFT銆佷笉鎵?Benchmark銆佷笉杩愯 Reserved Test銆佷笉 commit銆?> 鐪熷疄 Qwen3.5-9B Base 涓庢棦鏈?`/plan` 鏈嶅姟缁х画澶嶇敤锛汸14-C0 鍘熺粨鏋溿€佸悇杞慨澶嶅皾璇曞潎淇濈暀涓嶈鐩栥€?
## 0. 缁撹

鍦?*鍚屼竴鍐荤粨妯″瀷/Prompt/瑙ｇ爜鍙傛暟/宸ュ叿棰勭畻**涓嬶紝涓烘柊 Dev 20 澧炲姞鏈€灏?**Product Type/Role 鏍￠獙**骞朵慨澶?Gate retry 缁堟€併€傜粨鏋滐細

| 鎸囨爣锛圖ev 20锛?| Rule 淇鍓?| Rule 淇鍚?| Qwen 淇鍓?| Qwen 淇鍚?|
| --- | ---: | ---: | ---: | ---: |
| recommend 浠诲姟 FINALIZE | 15/15 | **15/15** | 15/15 | **13/15** |
| Top-1 鍛戒腑鏍囨敞鍚堟牸鍟嗗搧 | 6/15 | **13/15** | 8/15 | **13/15** |
| 鍟嗗搧绫诲瀷閿欒 | **9** | **0** | **7** | **0** |
| 闈炴帹鑽愪换鍔￠敊璇?FINALIZE | 0/5 | 0/5 | 0/5 | 0/5 |
| 涓嶅繀瑕?ASK锛坮ecommend锛?| 0 | **0** | 0 | 1锛?051锛?|
| 涓嶅繀瑕佹嫆缁濓紙recommend锛?| 0 | **0** | 0 | 1锛?050锛?|
| 姝ｇ‘ ASK锛坈larification 浠诲姟锛?| 4/5 | 4/5 | 4/5 | 4/5 |
| 闈炴硶鍔ㄤ綔/澶辫触 | 0 | 0 | 1锛堥潪娉?ASK锛?| 1锛?050 浠诲姟澶栵紝0060 闈炴硶 ASK 琚嫆锛?|
| 鐪熷疄妯″瀷璋冪敤/Token | 鈥?| 鈥?| 67 / 47,872+5,770 | **69 / 46,626+5,863** |
| 骞冲潎/鏈€澶фā鍨嬪欢杩?| 鈥?| 鈥?| 3,905.6 / 7,308.1 ms | **3,837.5 / 6,393.2 ms** |

**绫诲瀷閿欒锛?+7 鈫?0+0锛涗袱瀹舵纭?Top-1 閮芥彁鍗囧埌 13/15锛屼笖鏈嚭鐜扳€滃叏閮ㄦ嫆缁濇崲闆剁被鍨嬮敊璇€濄€?*
Rule 淇濈暀 2 鏉″悓绫诲瀷浣嗗亸濂?鎺掑簭閿欒鐨勬渚嬶紙0046銆?053锛夛紱Qwen 淇濈暀 2 鏉?planner 琛屼负閿欒 + 1 鏉￠潪娉?ASK锛?050/0051/0060锛夛紝鍧囧瀹炴姤鍛婏紝涓嶇敤淇敼鏍囩鎺╃洊銆?
## 1. 淇鍓嶈瘖鏂細閿欒褰掑洜锛堝彧璇?P14-C0 杞ㄨ抗锛?
`runs/p14c0r/diagnostics/`锛?
| 閿欒褰掑洜 | Rule锛? 鏉￠敊璇級 | Qwen锛? 鏉￠敊璇級 |
| --- | ---: | ---: |
| BM25 hit union 涓湁鈥滄纭被鍨?verified 鍊欓€夆€濓紝浣嗛€変簡閿欒绫诲瀷锛堟帓搴?Gate 鏀捐锛?| **8** | **5** |
| BM25 hit union 涓病鏈夆€滄纭被鍨?verified 鍊欓€夆€濓紙鍙洖/鍙揪鎬э級 | 1锛?046锛?| 2锛?050/0051锛?|
| 閫変腑鍟嗗搧绫诲瀷纭疄 UNKNOWN锛堣瘉鎹笉瓒筹級 | 0 | 0 |

缁撹锛氱粷澶ч儴鍒嗛敊璇笉鏄彫鍥炵己澶憋紝鑰屾槸 **Scorer 鎶婇敊璇被鍨嬫帓鍒板墠闈€丟ate 娌℃湁绫诲瀷/瑙掕壊绾︽潫鑰屾斁琛?*锛涘彧鏈夊皯閲忔槸鍊欓€夐噷纭疄娌℃湁姝ｇ‘绫诲瀷銆備慨澶嶅繀椤诲悓鏃惰鐩栤€滅被鍨嬮敊璇姝?FINALIZE鈥濆拰鈥滄棤姝ｇ‘绫诲瀷鍊欓€夋椂棰勭畻鍐呴噸妫€绱?鏈夌晫缁堟鈥濄€?
## 2. 鏈€灏忓疄鐜?
### 2.1 Product Type/Role 妯″潡

鏂板 `recommendation/preference/product_role.py`锛堟棤 evaluator 鏍囩銆佹棤 task_id 鐗瑰垽锛夛細

- 浠?**agent-visible query** 鎺ㄦ柇鏄惧紡鐩爣锛歚headphones / earbuds / keyboard / mouse / speaker`锛?- 鍖哄垎 target role锛歚main` / `accessory`锛堜緥濡?鈥渒eyboard stand鈥?鏄庣‘瑕侀厤浠讹紱鏃犳樉寮忕被鍨?鈫?`NOT_REQUIRED`锛屼笉鍒堕€犵害鏉燂級锛?- 鍊欓€夎鑹诧細`main / accessory / complement / substitute / unknown`锛?- 璇佹嵁鏉ユ簮锛歝atalogue title锛堢湡瀹?metadata锛夛細
  - 绫诲瀷璇嶅湪閰嶄欢璇嶄箣鍓嶄笖鏃犻厤浠朵慨楗?鈫?涓诲晢鍝侊紱
  - 鈥淢onitor Stand Riser 鈥?Keyboard storage鈥?杩欑被閰嶄欢璇嶅湪绫诲瀷璇嶅墠 鈫?accessory/complement锛?  - 鈥淜eyboard Stand for Desk鈥?鈫?accessory锛?  - 鈥淐ase for Headphones鈥?鈫?accessory锛?  - 鈥淲ireless Earbuds with Charging Case鈥?鈫?main锛堝甫閰嶄欢鐨勪富鍟嗗搧锛夛紱
- 杈撳嚭 `MATCH / MISMATCH / UNKNOWN / NOT_REQUIRED` + 绫诲瀷銆佽鑹层€佷緷鎹紱
- 瑙勫垯锛氱敤鎴疯涓诲晢鍝侊紝涓诲晢鍝佺被鍨嬩笉涓€鑷?鈫?MISMATCH锛涚敤鎴疯涓诲晢鍝佷絾鍊欓€夋槸閰嶄欢/浜掕ˉ鍝?鈫?MISMATCH锛涚敤鎴锋槑纭閰嶄欢鏃讹紝鍚堟牸閰嶄欢 MATCH銆佷富鍟嗗搧 MISMATCH锛涙棤绫诲瀷璇佹嵁 鈫?UNKNOWN锛堜笉鍐掑厖 MATCH锛夛紱query 鏃犳槑纭被鍨?鈫?NOT_REQUIRED銆?
### 2.2 Gate / Executor 鎺ュ叆

- `DecisionGate` 澧炲姞 `expected_product_type/expected_product_role`锛堥粯璁ょ┖锛屾棫璋冪敤涓嶆敼鍙橈級锛?- 鏈夋樉寮忕被鍨嬫椂锛屽€欓€夊繀椤?`role_status in {MATCH, NOT_REQUIRED}` 鎵嶅彲 `presentable`锛涚‖绾︽潫杩濊浠嶄紭鍏堢姝紱UNKNOWN 涓嶅綋浣滄弧瓒筹紱
- 鍙湁绫诲瀷閿欒/UNKNOWN 鍊欓€夋椂锛氶绠楀唴璇锋眰 `NEED_MORE_PRODUCT_EVIDENCE`/`NEED_MORE_CANDIDATES`锛涢绠楄€楀敖 鈫?`NO_VERIFIED_CANDIDATE_IN_SCOPE`锛坆ounded锛屼笉鏂█鍏ㄧ洰褰曟病鏈夛級锛?- `ElectronicsPlannerExecutor` 灏嗚鑹蹭笉鍖归厤鍊欓€変粠 Planner/FINALIZE 鍙鍊欓€夐泦涓繃婊わ紙ledger 浠嶄繚鐣欏叏閮ㄥ€欓€夊強鏉ユ簮锛屼笉涓㈡潵婧愶級锛屽苟淇濇寔 P14-B0 Scorer 鏉冮噸涓嶅彉锛?- 鏃犲悎鏍艰鑹插€欓€夋椂锛氬厛鍋?*涓€娆￠绠楀唴閲嶆绱?*锛涗粛鏃?鈫?鏈夌晫缁堟 `no_verified_candidate_in_scope`锛?- Gate retry 缁堟€侊細棰勭畻鑰楀敖鍚庝笉鍐嶈繑鍥為潪缁堟€?`need_more_*`锛汦xecutor/harness 灏嗗叾鏄犲皠涓?`budget_exhausted` / `no_verified_candidate_in_scope`锛屼笉浼氭棤闄愬惊鐜紱鏈В鍐?ASK 浠嶄笉鑳?FINALIZE锛?- Rule/Qwen 璧?*鍚屼竴涓?executor銆佸悓涓€涓?Gate銆佸悓涓€浠?role 瑙勫垯銆佸悓涓€棰勭畻**銆?
## 3. 淇鍚庣粨鏋滐紙鏂?Dev 20锛涘悓涓€鍐荤粨閰嶇疆锛?
閫愪换鍔★細`runs/p14c0r/rule_dev/trajectories/`銆乣runs/p14c0r/qwen_dev/trajectories/`銆乣runs/p14c0r/comparison.json`銆乣runs/p14c0r/error_analysis.jsonl`銆?
### Rule

- 15/15 finalized锛?3/15 correct锛?*type mismatch 0**锛沠alse ask/refusal 0锛沨ard violations 0锛沜orrect clarification 4/5锛沜orrect unknown termination 1/1锛沞rrors 0銆?- 2 鏉″墿浣欓敊璇潎涓?*鍚岀被鍨嬪晢鍝佷絾鍋忓ソ/鎺掑簭涓嶅尮閰?*锛?046 keyboard銆?053 keyboard锛夛紝涓嶅啀鏄被鍨嬮棶棰樸€?
### Qwen3.5-9B Base

- finalized 13/15锛?3/15 correct锛?*type mismatch 0**锛沠alse ask 1锛?051锛夈€乫alse refusal 1锛?050锛夈€乪rror 1锛?060锛歚ASK requires an ambiguity/need signal`锛岃 contract 鎷掔粷锛岄潪浼€犳垚鍔燂級锛沨ard violations 0锛沠alse non-recommend finalize 0/5锛沜orrect clarification 4/5銆?- 69 娆＄湡瀹炶皟鐢紱46,626 input / 5,863 output tokens锛涘钩鍧?3,837.5 ms / 鏈€澶?6,393.2 ms銆?- 鐩告瘮淇鍓嶏細correct 8鈫?3锛宼ype error 7鈫?锛涗粛鏈?2 鏉¤瑙掕壊/planner 鑱斿悎璺緞褰卞搷锛?050/0051锛夛紝灞炰簬 Planner 鍐崇瓥/ASK 琛屼负鑰岄潪绫诲瀷鏀捐銆?
### 棰濆妫€绱㈡垚鏈?
| | 淇鍓?search calls | 淇鍚?search calls |
| --- | ---: | ---: |
| Rule | 20 | 20锛? 棰濆锛?|
| Qwen | 21 | 22锛?1 娆★紱浠呭湪瑙掕壊閲嶆绱㈣矾寰勮Е鍙戯級 |

Qwen 妯″瀷璋冪敤 67鈫?9锛?2锛夛紝涓昏鏉ヨ嚜瑙掕壊杩囨护鍚庣殑閲嶆绱?璇佹嵁杞紱Token/寤惰繜瑙佷笂琛ㄣ€?
## 4. Gate Retry 缁堟€佷慨澶?
- 鍥哄畾 Dev 20 鐨勬墍鏈変慨澶嶅悗杞ㄨ抗涓?*涓嶅瓨鍦?* `need_more_candidates` / `need_more_product_evidence` / 鍗婇€旂姸鎬侊紱缁堟€佷负 `finalized` / `awaiting_clarification` / `no_verified_candidate_in_scope` / `budget_exhausted` / `error`锛?- 棰勭畻鑰楀敖鍚庝笉鍐嶆棤闄?RETRIEVE锛涙病鏈夊０绉扳€滃叏鐩綍娌℃湁鍚堟牸鍟嗗搧鈥濓紙`bounded_scope=True`锛夛紱ASK 涓嶈浼€犲洖澶嶏紱0050 涓烘湁鐣屾嫆缁濄€?051 涓鸿 ASK锛堜繚鐣欎负澶辫触妗堜緥锛夈€?
## 5. 澶辫触妗堜緥锛堜繚鐣欙級

1. **Rule 0046 / 0053**锛氱被鍨嬫纭絾鍋忓ソ/鎺掑簭閿欒锛? 鏉★紝浠嶈鍏?Top-1 鏈懡涓級銆?2. **Qwen 0050**锛氳鑹茶繃婊?閲嶆绱㈠悗杩涘叆鏈夌晫鎷掔粷锛堜笉蹇呰鎷掔粷 1锛夈€?3. **Qwen 0051**锛氬湪 recommend 浠诲姟涓婂彂鍑轰笉蹇呰 ASK锛?锛夈€?4. **Qwen 0060**锛坣o_verified 闈炴帹鑽愪换鍔★級锛氭ā鍨嬭姹傜敤鎴锋緞娓咃紝琚?`run_planner` 鎷掔粷锛圥lannerContractError锛夛紝璁板綍涓?error锛涙病鏈夐敊璇?FINALIZE銆?5. Rule 鐨?0046 鍟嗗搧绫诲瀷鏍囨敞鐢?catalogue title 鎺ㄦ柇锛堚€淢onitor Stand鈥eyboard storage鈥濓級锛屾垜浠凡鐢?role 鐨勯厤浠惰瘝浼樺厛瑙勫垯灏嗗叾浠庡€欓€夎繃婊わ紱浠嶅彲鑳芥湁涓埆鐩綍 title 璇箟妯＄硦妗堜緥锛屽睘浜庤瘉鎹笉瓒宠€岄潪璇勫垎閿欒銆?
## 6. 娴嬭瘯

| 娴嬭瘯 | 缁撴灉 |
| --- | --- |
| `tests/test_p14c0r_product_role.py` | 9 椤癸細涓?閰嶄欢鍖哄垎銆佺被鍨?MATCH/MISMATCH/UNKNOWN/NOT_REQUIRED銆丟ate 绂佹绫诲瀷閿欒銆佹棤鏄惧紡绫诲瀷涓嶅埗閫犵害鏉熴€佹湁鐣?retry銆佷慨澶嶅悗闆?type error銆佹棫缁撴灉淇濈暀銆丷eserved 闆惰Е纰?|
| 瀹氬悜鍥炲綊锛圥14-C0 contract + P14-B1 Gate + P14-C0-R锛?| **31 passed**锛?1.0 s锛?|
| 鏈繍琛?| 鍏ㄤ粨搴?pytest銆丷eserved Test銆佽缁?SFT |

## 7. 鍒ゅ畾

| 鍒ゅ畾 | 缁撴灉 | 璇存槑 |
| --- | --- | --- |
| `PRODUCT_ROLE_GATE` | **PASS** | 绫诲瀷 MATCH/MISMATCH/UNKNOWN銆佷富鍟嗗搧/閰嶄欢/浜掕ˉ/鏇夸唬鍖哄垎锛涚被鍨嬮敊璇笉鑳?FINALIZE |
| `RULE_QWEN_SAME_IMPLEMENTATION` | **PASS** | 鍚屼竴 executor/Gate/role 瑙勫垯/棰勭畻锛涗粎 planner 涓嶅悓 |
| `GATE_RETRY_TERMINAL` | **PASS** | 棰勭畻鑰楀敖鏈夌晫缁堟锛涙棤鏃犻檺寰幆/闈炵粓鎬佹畫鐣?|
| `RULE_COMPARISON` | **PASS** | 淇鍓嶅悗鍚?Dev 20 瀵圭収锛汻ule 13/15銆丵wen 13/15 |
| `EVALUATION_ISOLATION` | **PASS** | 鏃犳爣绛炬硠婕忥紱Reserved Test 鏈繍琛岋紱鏃?P14-C0 缁撴灉鏈鐩?|
| `LEGACY_REGRESSION` | **PASS** | 31 椤瑰畾鍚戞祴璇曢€氳繃锛汼corer 鏉冮噸/鏃?Serving/Planner 榛樿鍔ㄤ綔鏈敼 |
| **P14-C0-R** | **GO** | Product Type/Role 鏈€灏忎慨澶嶈揪鎴愮洰鏍囷細绫诲瀷閿欒 16鈫?锛屾纭?Top-1 18鈫?6/30锛屾甯告帹鑽愭湭琚繃搴﹂樆鏂紱淇濈暀 Qwen planner 娈嬪樊涓庡紑鍙戦泦璇婃柇璇存槑 |

> 鏈粨鏋滃睘浜?new Dev 鐨勫紑鍙戦泦璇婃柇锛圥14-C0 Dev 宸插弬涓庨棶棰樺彂鐜帮級锛屼笉鑳界О涓哄叏鏂扮嫭绔嬫祴璇曪紱鏍囩浠?provisional銆俁eserved Test 鏈繍琛岋紝涓嶈兘鎹瀹ｇО娉涘寲浼樺娍銆?
## 8. 浜х墿

- `runs/p14c0r/comparison.json`锛堜慨澶嶅墠鍚庛€侀€愪换鍔°€佺被鍨嬮敊璇鏁般€丵wen token/latency锛?- `runs/p14c0r/rule_dev/`銆乣runs/p14c0r/qwen_dev/`锛堢湡瀹炶建杩癸級
- `runs/p14c0r/diagnostics/`锛堜慨澶嶅墠绫诲瀷閿欒褰掑洜锛氬彫鍥?vs 鎺掑簭/Gate锛?- `runs/p14c0r/error_analysis.jsonl`
- 淇灏濊瘯淇濈暀锛歚runs/p14c0r_first_attempt/`銆乣runs/p14c0r_second_attempt/`銆乣runs/p14c0r_third_attempt/`锛堝潎鏈鐩栵級
- 鍘?P14-C0锛歚runs/p14c0/` 鏈鐩栵紱鐪熷疄 Qwen 鏈嶅姟/闅ч亾浠嶅湪杩愯锛堟湭鍋氬疄渚嬬敓鍛藉懆鏈熸搷浣滐級
- 婧愮爜宸紓锛歔`P14C0R_SOURCE_DIFF.txt`](P14C0R_SOURCE_DIFF.txt)

