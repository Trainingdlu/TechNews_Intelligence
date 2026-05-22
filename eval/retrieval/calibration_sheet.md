# G2 Relevance Calibration Sheet (BLIND)

Total pairs to label: **100**

For each pair, judge how relevant the article is to the QUERY and fill `[ ]` with 0, 1, or 2:
- **2** = highly relevant (article is directly about what the query asks)
- **1** = partially relevant (mentions the topic/entity but not the focus)
- **0** = not relevant

Do NOT change the `pair_id` lines. Only fill the `Your score:` brackets.

---

## 1. `ret_02_anthropic_enterprise::https://techcrunch.com/2026/05/13/anthropic-courts-a-new-kind-of-customer-small-business-owners/`

**Query:** Anthropic 企业市场进展

**Title:** [商业] Anthropic推出小微企业专属服务拓展下沉市场

**Summary:** Anthropic推出“Claude小微企业版”服务，依托Claude Work平台提供账务处理、商业洞察及广告生成等功能，并已打通QuickBooks、Canva等主流软件接口。该策略旨在覆盖占美国GDP 44%的3600万家小微企业。相较于OpenAI 2023年推出的同类服务，Anthropic计划通过全美10城巡回及免费培训工作坊加速获客，标志着AI平台竞争重心正向下沉市场转移。

**Your score:** `[2 ]`

---

## 2. `ret_02_anthropic_enterprise::https://www.anthropic.com/news/gates-foundation-partnership`

**Query:** Anthropic 企业市场进展

**Title:** [商业] Anthropic与盖茨基金会达成2亿美元合作

**Summary:** Anthropic宣布与盖茨基金会达成4年期合作，承诺投入2亿美元用于医疗资助、Claude算力支持及技术援助。资金将重点投向三大领域：全球健康与生命科学方面，助力中低收入国家疫苗与疗法研发，并优化疟疾及结核病部署预测；教育领域，联合开发K-12数学辅导与基础读写算AI工具；经济流动方面，改进农业专项模型并构建职业技能认证与就业指导平台。该项目由公益部署团队主导，旨在拓展AI在公共服务领域的应用。

**Your score:** `[ 2]`

---

## 3. `ret_03_claude_updates::https://claude.com/blog/how-claude-code-works-in-large-codebases-best-practices-and-where-to-start`

**Query:** Claude 新功能更新

**Title:** [AI] Claude Code在大型代码库中的工作原理及最佳实践

**Summary:** 文章阐述Claude Code在大型企业代码库中的部署模式，指出其采用代理搜索替代传统RAG以避免索引延迟。核心架构依赖五大扩展层：CLAUDE.md提供分层上下文，钩子实现自动化改进，技能按需加载专业知识，插件分发标准化配置，MCP与LSP集成扩展外部工具与符号级导航。建议企业通过目录级初始化及设立专属AI工具管理角色来优化上下文效率与采用率。

**Your score:** `[2 ]`

---

## 4. `ret_03_claude_updates::https://claude.com/pricing`

**Query:** Claude 新功能更新

**Title:** [AI] Anthropic公布Claude订阅计划与API定价细则

**Summary:** Anthropic公布Claude产品订阅与API定价体系。订阅计划涵盖Pro（每月20美元）、Max（起价100美元）及企业版（按席位计费），集成Claude Code、办公生态插件与企业级安全管控。API定价方面，Opus 4.7输入输出分别为每百万标记5与25美元，Sonnet 4.6为3与15美元，Haiku 4.5为1与5美元。平台同步推出托管智能体、网页搜索及沙箱代码执行等按量计费模块，支撑从个人提效至企业规模化部署的全场景需求。

**Your score:** `[1 ]`

---

## 5. `ret_03_claude_updates::https://x.com/aaronp613/status/2049986504617820551`

**Query:** Claude 新功能更新

**Title:** [安全] 苹果Support应用更新意外遗留Claude.md配置文件

**Summary:** 苹果公司在Apple Support应用v5.13版本更新中，意外将内部Claude.md配置文件打包至公开发布端。该文件通常包含针对Anthropic AI模型的系统提示词或开发环境配置指令。此次疏漏属于应用打包流程中的配置泄露事件，虽未涉及核心用户隐私数据，但暴露出应用发布审查环节的管控漏洞，目前相关文件已被技术社区捕获并分析。

**Your score:** `[0 ]`

---

## 6. `ret_06_meta_llama::https://aws.amazon.com/blogs/machine-learning/agent-guided-workflows-to-accelerate-model-customization-in-amazon-sagemaker-ai/`

**Query:** Meta AI 与 Llama 模型

**Title:** [AI] Amazon SageMaker AI推出智能体引导工作流以加速模型定制

**Summary:** Amazon SageMaker AI现已集成智能体引导的模型定制工作流。开发者通过自然语言交互，驱动AI编码智能体调用模块化技能，自动化完成数据预处理、SFT或DPO及RLVR微调选型、评估及至Bedrock或SageMaker端点的部署全流程。该方案基于ACP协议接入JupyterLab环境，自动生成集成MLflow指标追踪的可复用代码，在降低Token消耗的同时将模型定制周期从数月缩短至数天。

**Your score:** `[ 0]`

---

## 7. `ret_09_deepseek::https://simonwillison.net/2026/Apr/24/deepseek-v4/`

**Query:** DeepSeek 模型与融资

**Title:** [AI] DeepSeek发布V4模型：性能接近前沿，定价大幅降低

**Summary:** 中国AI实验室DeepSeek发布V4系列预览模型，包含V4-Pro与V4-Flash。两款模型采用MoE架构，支持100万token上下文。定价极具竞争力：Flash输入/输出为每百万token 0.14/0.28美元，Pro为1.74/3.48美元，均显著低于同类前沿模型。架构优化使两者在百万级上下文下的FLOPs与KV缓存占用大幅降低。基准测试表明其推理能力接近行业顶尖水平，性能预计落后最新前沿模型3至6个月。

**Your score:** `[2 ]`

---

## 8. `ret_100_opensource_llm_en::https://dpc.pw/posts/i-dont-want-your-prs-anymore/`

**Query:** open source LLM releases

**Title:** [开发] 开源维护者直言不再需要你的PR：LLM时代协作模式的重构

**Summary:** 文章分析LLM普及对开源项目协作流程的影响，指出外部PR带来的安全风险、风格差异与沟通成本已超过其价值。随着大模型具备快速迭代能力，维护者在代码理解、架构设计与审查环节的瓶颈无法通过直接合并代码缓解。为此，作者建议社区贡献转向提供使用反馈、探讨设计方案、提交详尽缺陷报告、分享提示词原型、参与代码审查及自主Fork项目，以优化维护效率并推动定制化开发。

**Your score:** `[0 ]`

---

## 9. `ret_100_opensource_llm_en::https://news.ycombinator.com/item?id=48085993`

**Query:** open source LLM releases

**Title:** [开发] HN开发者月度项目分享（2026年5月）

**Summary:** 核心议题：独立开发者与工程师的5月项目进展分享。典型项目：涵盖基础设施（vLLM推理网关优化、Betterleaks漏洞扫描替代方案）、AI应用（AI辅助SOW撰写工具、动态生成资产的AI跑团模拟器Orpheus）、独立产品（Kagi平替搜索引擎Uruky、SQL可视化画布Kavla）及原生开发工具（非JS HTTP客户端、Linux任务管理器）。技术趋势：强调低成本算力优化、资产复用策略以及AI与垂直场景的深度结合，开发者普遍采用快速原型验证与开源协作模式推进。

**Your score:** `[ 0]`

---

## 10. `ret_100_opensource_llm_en::https://ollama.com/blog/mlx`

**Query:** open source LLM releases

**Title:** [AI] Ollama预览版现基于MLX支持Apple Silicon

**Summary:** Ollama 0.19预览版在Apple Silicon上通过MLX框架实现加速，支持Qwen3.5-35B-A3B模型的NVFP4量化格式。新版本优化了统一内存架构下的预填充与解码性能，并改进缓存机制以提升编码代理任务效率。测试显示M5系列芯片可显著提升首字延迟与生成速度。

**Your score:** `[2 ]`

---

## 11. `ret_100_opensource_llm_en::https://www.clojuriststogether.org/news/q2-2026-funding-announcement/`

**Query:** open source LLM releases

**Title:** [开发] Clojurists Together 公布 2026 年第二季度开源资助计划

**Summary:** Clojurists Together 宣布 2026 年 Q2 开源资助计划，总计拨款 3.1 万美元支持 5 个 Clojure 生态项目。核心开发方向包括：Malli（优化递归 Schema 验证的内存占用）、Uncomplicate AI（构建基于 ONNX 的本地 LLM 推理库）、SciCloj（完善数据分析与可视化库文档）、Gloat（提升 Clojure 至 Go 编译器的二进制体积与跨平台兼容性）及 PluMCP（适配 MCP 协议新规范以支持 Agentic AI 架构）。

**Your score:** `[ 1]`

---

## 12. `ret_100_opensource_llm_en::https://z.ai/blog/glm-5.1`

**Query:** open source LLM releases

**Title:** [AI] Z.ai发布GLM-5.1：面向长程任务的下一代智能体模型

**Summary:** Z.ai团队发布新一代智能体编程模型GLM-5.1，主打长程任务优化与自主迭代。模型在SWE-Bench Pro取得58.4分，显著领先前代GLM-5。实测中，该模型在超600次迭代中将向量数据库查询性能提升至21.5k QPS，在GPU内核优化任务实现3.6倍加速，并能在8小时内持续完善复杂Linux桌面网页应用。GLM-5.1已基于MIT协议开源，权重文件上线HuggingFace及ModelScope，并接入Claude Code等主流编程代理工具。

**Your score:** `[2 ]`

---

## 13. `ret_101_agent_frameworks_en::https://github.com/facebookresearch/hyperagents`

**Query:** AI agent frameworks and tools

**Title:** [AI] HyperAgents：自指式自我改进智能体

**Summary:** Meta发布HyperAgents框架，支持智能体通过自指机制优化任意可计算任务。项目包含任务代理与元代理核心模块，依赖多模态API（OpenAI/Anthropic/Gemini）。实验日志以分卷压缩包形式存储，附带安全警告提示执行模型生成代码存在潜在风险。

**Your score:** `[ 2]`

---

## 14. `ret_102_datacenter_energy_en::https://techcrunch.com/2026/04/27/data-center-demand-drives-66-surge-in-natural-gas-power-plant-costs/`

**Query:** data center energy demand for AI

**Title:** [硬件] 数据中心需求推升天然气发电厂建设成本66%

**Summary:** 受数据中心电力需求推升，新建天然气发电厂成本两年内上涨66%，单位造价从2023年不足1500美元每千瓦升至2157美元，建设周期延长23%。预计至2035年相关电力需求将达106吉瓦，导致燃气轮机供应紧张，设备价格较2019年上涨195%，交付排期延至2030年代初。受此影响，谷歌已转向可再生能源搭配长时储能技术的供电方案。

**Your score:** `[2 ]`

---

## 15. `ret_103_enterprise_adoption_en::https://blog.google/company-news/outreach-and-initiatives/small-business/ai-for-small-businesses/`

**Query:** enterprise AI adoption

**Title:** [AI] 谷歌推出AI工具及专属优惠助力小型企业增长

**Summary:** 谷歌宣布将AI能力全面接入小企业工作流，推出Gemini企业版应用与Workspace深度集成，支持构建AI智能体以自动化处理销售数据与客户会议记录。同期配合全国小企业周推出多项优惠，包括Workspace首三个月九五折、Gemini企业版30天免费试用及最高6000美元广告抵扣额。此外，提供Pomelli与Nano Banana等AI设计工具快速生成营销物料，并通过搜索、地图与YouTube的AI算法优化客户触达效率，配套免费AI技能培训课程。

**Your score:** `[2 ]`

---

## 16. `ret_106_openai_enterprise::https://techcrunch.com/2026/05/15/openai-launches-chatgpt-for-personal-finance-will-let-you-connect-bank-accounts/`

**Query:** OpenAI 的企业级产品

**Title:** [AI] OpenAI推出ChatGPT个人理财功能，支持连接银行账户

**Summary:** OpenAI面向美国Pro用户推出ChatGPT个人理财预览功能，通过集成Plaid支持连接超12000家金融机构。该功能依托GPT-5.5模型增强上下文推理能力，并整合此前收购的Hiro团队技术，目前每月超2亿用户已频繁咨询财务问题。工具已上线Web与iOS端，提供账户管理及断开连接后30天数据清理机制。

**Your score:** `[2 ]`

---

## 17. `ret_107_nvidia_gpu_arch::https://developer.nvidia.com/blog/nvidia-platform-delivers-lowest-token-cost-enabled-by-extreme-co-design/`

**Query:** NVIDIA 新一代 GPU 架构

**Title:** [AI] NVIDIA平台凭借极致协同设计实现最低Token成本

**Summary:** NVIDIA在MLPerf推理v6.0基准测试中公布最新成绩，其Blackwell Ultra平台结合TensorRT-LLM与Dynamo软件栈，在DeepSeek-R1等模型推理中取得最高吞吐量，单节点性能较此前提升达二点七倍。该成果验证了软硬件协同设计在降低生成式AI Token成本与满足高交互部署场景中的核心优势。

**Your score:** `[2 ]`

---

## 18. `ret_108_aws_model_hosting::https://aws.amazon.com/blogs/machine-learning/introducing-claude-platform-on-aws-anthropics-native-platform-through-your-aws-account/`

**Query:** AWS 的大模型托管服务

**Title:** [AI] AWS正式上线Anthropic原生Claude平台，实现统一鉴权与计费管理

**Summary:** AWS宣布Claude Platform正式集成至AWS生态，企业可通过现有账户直接调用Anthropic原生API及智能体功能。该服务数据请求处理于外部架构边界，全面兼容AWS IAM鉴权、Marketplace计费与CloudTrail审计。服务覆盖全球多区域，支持通过Workspace隔离环境，并允许使用Anthropic SDK或Claude Code等客户端快速接入，旨在简化企业级AI工作流的部署与成本管控。

**Your score:** `[ 2]`

---

## 19. `ret_108_aws_model_hosting::https://stratechery.com/2026/an-interview-with-openai-ceo-sam-altman-and-aws-ceo-matt-garman-about-bedrock-managed-agents/`

**Query:** AWS 的大模型托管服务

**Title:** [AI] OpenAI模型登陆AWS Bedrock，双方联合推出托管智能体服务

**Summary:** 微软与OpenAI修订合作协议，微软独家授权转为非独家，OpenAI产品全面开放跨云部署。亚马逊AWS同步推出Bedrock托管智能体服务，将OpenAI前沿模型与AWS IAM权限、VPC网络及审计日志深度集成。数据全程留存于客户AWS环境，底层推理算力混合调用Trainium与GPU，运维由AWS一线承接。该方案旨在降低企业级智能体开发门槛。

**Your score:** `[ 2]`

---

## 20. `ret_109_meta_opensource_tools::https://techcrunch.com/2026/05/12/threads-tests-a-meta-ai-integration-that-works-similarly-to-grok/`

**Query:** Meta 的开源 AI 工具

**Title:** [AI] Threads 测试集成 Meta AI 功能

**Summary:** Threads 正在马来西亚、沙特阿拉伯、墨西哥、阿根廷和新加坡测试 Meta AI 集成功能。用户可通过提及 @meta.ai 获取实时趋势与新闻解读，AI 将以公开回复形式回应。该功能旨在对标 X 平台的 Grok，Meta 声称其内置更强的安全护栏。公司表示将根据早期反馈优化体验，后续逐步扩大开放范围。

**Your score:** `[ 0]`

---

## 21. `ret_109_meta_opensource_tools::https://www.businessinsider.com/meta-new-ai-tool-tracks-staff-activity-sparks-concern-2026-4`

**Query:** Meta 的开源 AI 工具

**Title:** [AI] Meta内部AI训练工具强制收集员工操作数据引发争议

**Summary:** Meta在美国区员工终端部署内部AI训练工具，强制捕获键盘输入、鼠标轨迹及指定办公应用（如Gmail、VSCode）的屏幕内容，用于提升AI代理的图形界面交互能力。该程序不可选择退出，仅作用于工作设备。内部反馈显示员工集中关注数据隐私与退出机制。Meta回应称已配置隐私过滤机制，采集数据仅限模型训练，且工作设备监控符合既有员工协议。

**Your score:** `[ 0]`

---

## 22. `ret_109_meta_opensource_tools::https://www.reuters.com/sustainability/boards-policy-regulation/meta-start-capturing-employee-mouse-movements-keystrokes-ai-training-data-2026-04-21/`

**Query:** Meta 的开源 AI 工具

**Title:** [商业] Meta推行员工操作数据采集计划以训练AI并启动全球裁员

**Summary:** Meta宣布部署MCI追踪软件，采集美国员工鼠标轨迹与按键数据以训练自主AI智能体。数据仅限模型优化，不用于绩效考核。该计划隶属于Agent Transformation Accelerator架构，配合内部岗位重组。公司定于5月20日启动10%全球裁员，并计划年内追加削减。该监控举措引发数据隐私争议，欧美在劳动监管与GDPR合规层面存在显著法律差异。

**Your score:** `[ 0]`

---

## 23. `ret_110_apple_partnership::https://techcrunch.com/2026/04/27/openai-ends-microsoft-legal-peril-over-its-50b-amazon-deal/`

**Query:** Apple 与 OpenAI 或谷歌的 AI 合作

**Title:** [商业] OpenAI与微软重订合作协议，化解500亿美元亚马逊交易法律风险

**Summary:** 微软与OpenAI宣布重新协商合作协议，终止微软对OpenAI产品及知识产权的独家授权，改为截至2032年的非独家许可。该条款解除OpenAI此前与亚马逊高达500亿美元投资协议引发的法律纠纷。新协议确立微软至2032年仍为OpenAI主要云合作伙伴，产品优先在Azure上线。财务条款方面，微软停止向OpenAI支付收入分成，而OpenAI向微软支付分成的条款延续至2030年并增设上限。微软目前仍持有OpenAI约27%的股份。

**Your score:** `[ 2]`

---

## 24. `ret_111_ai_infra_investment::https://static1.squarespace.com/static/50363cf324ac8e905e7df861/t/6a0af5d0484fbf5fe9a7743e/1779103184855/2026-Spring-AI.pdf`

**Query:** AI 基础设施投资

**Title:** [AI] AI重塑技术平台与资本部署格局

**Summary:** 报告将生成式AI定位为新一轮技术平台迁移，指出2026年四大科技巨头资本支出计划达7000亿美元，重点投向数据中心与算力芯片。当前面临电力与硬件供应链瓶颈，且前沿模型正趋于同质化。企业部署呈现探索性特征，初期聚焦代码生成、客服与后台自动化。长期看，AI算力或演变为低毛利基础设施，商业价值与创新重心将向应用层转移，定价模式有望转向公用事业化计量。

**Your score:** `[ 2]`

---

## 25. `ret_113_ai_finance::https://blog.google/company-news/outreach-and-initiatives/creating-opportunity/ai-economy-forum/`

**Query:** AI 在金融领域的应用

**Title:** [生态] 谷歌联合MIT举办AI经济论坛，宣布研究投资与劳动力转型计划

**Summary:** 谷歌在华盛顿联合MIT举办首届AI经济论坛，聚焦AI对宏观经济与就业市场的影响。公司宣布依托1.2亿美元全球AI机会基金及前期10亿美元教育投入，深化AI经济研究并扩展职业培训网络。目前已与强生基金会、Jobs for the Future及美国制造研究所达成合作，定向推进乡村医疗人员AI扫盲、百家企业学徒项目扩展及4万名制造员工技能升级。谷歌同步支持专项政策法案，旨在通过政企协同平稳推进产业数字化转型。

**Your score:** `[ 0]`

---

## 26. `ret_114_ai_talent::https://techcrunch.com/2026/03/30/ai-work-boss-supervisor-us-quinnipiac-poll/`

**Query:** AI 人才争夺与挖角

**Title:** [AI] 15%美国民众愿接受AI担任直属上司

**Summary:** 据昆尼皮亚克大学2026年3月民调，15%的美国人愿接受由AI程序分配任务与排班作为直接主管。调查涵盖1,397名成年人，70%受访者认为AI发展将减少人类就业机会，30%在职者担忧自身岗位被AI取代。

**Your score:** `[2 ]`

---

## 27. `ret_114_ai_talent::https://www.citadelsecurities.com/news-and-insights/2026-global-intelligence-crisis/`

**Query:** AI 人才争夺与挖角

**Title:** [AI] 软件工程师岗位需求攀升与AI就业市场分析

**Summary:** 美国软件工程师岗位发布量同比上涨11%，AI资本支出占GDP比重达2%。圣路易斯联储数据显示，生成式AI工作使用率保持稳定，未见指数级扩散迹象。技术扩散受算力成本与组织整合限制呈现S型曲线，自动化替代存在明确经济边界。宏观分析指出，AI属正向供给冲击，数据中心建设已带动招聘。综合历史规律，AI更可能作为劳动力补充以抵消增长阻力，而非引发大规模失业。

**Your score:** `[ 2]`

---

## 28. `ret_115_context_length::https://developer.nvidia.com/blog/minimax-m2-7-advances-scalable-agentic-workflows-on-nvidia-platforms-for-complex-ai-applications/`

**Query:** 大模型上下文长度进展

**Title:** [AI] MiniMax M2.7在NVIDIA平台上推进可扩展智能体工作流

**Summary:** MiniMax开源了230B参数且单Token激活10B的M2.7 MoE模型，专为复杂智能体任务优化。该模型深度适配NVIDIA Blackwell Ultra GPU，通过与vLLM和SGLang框架集成QK RMS Norm与FP8 MoE定制内核，在1K上下文测试集上吞吐量分别最高提升2.5倍和2.7倍。开发者可借助NemoClaw一键部署智能体环境，或利用NeMo框架进行强化学习与微调，实现高效企业级落地。

**Your score:** `[ 1]`

---

## 29. `ret_116_inference_chip_startup::https://darkbloom.dev`

**Query:** AI 推理芯片创业公司

**Title:** [AI] Darkbloom：基于闲置Mac的去中心化隐私AI推理网络

**Summary:** Eigen Labs推出Darkbloom去中心化推理网络，利用逾1亿台闲置Apple Silicon设备构建AI算力池。平台通过端到端密文路由、Apple安全隔区密钥绑定及OS级运行时锁定，实现硬件级数据防窥。接口完全兼容OpenAI标准，支持文本、图像与语音模态，推理报价较中心化平台下降50%，硬件贡献方获得100%收益。

**Your score:** `[0 ]`

---

## 30. `ret_116_inference_chip_startup::https://techcrunch.com/2026/05/06/deepseek-could-hit-45b-valuation-from-its-first-investment-round/`

**Query:** AI 推理芯片创业公司

**Title:** [商业] DeepSeek首轮融资估值或达450亿美元

**Summary:** 中国AI实验室DeepSeek正进行首轮外部融资，估值在数周内由200亿美元跃升至450亿美元。本轮由国家集成电路产业投资基金牵头，腾讯与阿里巴巴亦在洽谈参与。创始人梁文锋持股近90%，此前未寻求外部投资。本轮融资旨在通过向员工授股应对人才流失，并依托华为芯片生态加速国产AI技术自主化，以规避美国出口管制。

**Your score:** `[ 0]`

---

## 31. `ret_117_ai_copyright::https://www.404media.co/pinterest-is-drowning-in-a-sea-of-ai-slop-and-auto-moderation/`

**Query:** AI 内容生成与版权争议

**Title:** [AI] Pinterest因AI内容泛滥和自动审核遭用户批评

**Summary:** Pinterest因大量AI生成内容涌入和自动审核系统误判引发用户强烈不满。艺术家称其手绘作品被错误标记为AI生成，且申诉流程繁琐低效。社区反馈AI内容充斥信息流，影响平台创作生态。

**Your score:** `[ 2]`

---

## 32. `ret_118_google_tpu::https://techcrunch.com/2026/05/07/google-unveils-whoop-like-screenless-fitbit-air/`

**Query:** 谷歌 TPU 与自研芯片

**Title:** [硬件] 谷歌发布无屏幕可穿戴设备Fitbit Air

**Summary:** 谷歌发布售价100美元的无屏幕可穿戴设备Fitbit Air，主打全天候佩戴与轻量化设计。该设备重量仅12克（含表带），体积较前代缩小25%至50%，支持七天续航及5分钟快充，具备50米防水能力。内置心率、血氧及睡眠等生理指标监测功能，并可联动Pixel Watch使用。同步上线的Google Health应用引入Gemini驱动的AI健康教练，为订阅用户提供定制训练与数据分析服务。设备定于5月26日发售。

**Your score:** `[ 0]`

---

## 33. `ret_119_openai_compute::https://stratechery.com/2026/an-interview-with-openai-ceo-sam-altman-and-aws-ceo-matt-garman-about-bedrock-managed-agents/`

**Query:** OpenAI 的算力与数据中心

**Title:** [AI] OpenAI模型登陆AWS Bedrock，双方联合推出托管智能体服务

**Summary:** 微软与OpenAI修订合作协议，微软独家授权转为非独家，OpenAI产品全面开放跨云部署。亚马逊AWS同步推出Bedrock托管智能体服务，将OpenAI前沿模型与AWS IAM权限、VPC网络及审计日志深度集成。数据全程留存于客户AWS环境，底层推理算力混合调用Trainium与GPU，运维由AWS一线承接。该方案旨在降低企业级智能体开发门槛。

**Your score:** `[ 2]`

---

## 34. `ret_11_opensource_llm::https://developers.openai.com/api/docs/changelog`

**Query:** 开源大模型最新发布

**Title:** [AI] OpenAI在API中正式发布GPT-5.5及GPT-5.5 Pro模型

**Summary:** OpenAI于2026年4月通过API发布GPT-5.5及GPT-5.5 Pro。新模型支持100万上下文窗口、图像输入、内置计算机控制、MCP协议与网络搜索，默认采用中等推理强度；Pro版面向高算力复杂场景。同期上线GPT Image 2与Sora-2视频模型，新增批量API与1080p输出。智能体开发套件同步引入沙盒环境与记忆管理功能，全面强化多模态交互与自动化工作流支持。

**Your score:** `[ 0]`

---

## 35. `ret_11_opensource_llm::https://simonwillison.net/2026/May/19/5-minute-llms/`

**Query:** 开源大模型最新发布

**Title:** [AI] 过去六个月LLM发展回顾：编码智能体成熟与本地模型崛起

**Summary:** 过去六个月LLM领域呈现两大趋势。编码智能体经强化学习优化，已由实验性工具转为可日常使用的开发主力。个人AI助手“Claws”类项目快速兴起，带动边缘计算硬件需求。开源侧模型能力显著增强，Gemma 4系列与1.5TB参数开源模型GLM-5.1相继发布。整体来看，前沿模型迭代加速，本地可部署模型的实际性能已超出预期，推动AI应用向开发辅助与个人助理场景落地。

**Your score:** `[ 0]`

---

## 36. `ret_120_hallucination::https://aphyr.com/posts/419-the-future-of-everything-is-lies-i-guess-new-jobs`

**Query:** 大模型幻觉与可靠性

**Title:** [AI] 探讨大模型普及催生的新型人机协作岗位

**Summary:** 文章探讨大模型规模化部署催生的新型人机协作岗位。针对LLM输出的不确定性与幻觉问题，提出设立提示词专员负责优化模型输入。流程与统计工程师将负责构建质量控制体系及测量模型偏差。行业专家转型为模型训练师以提供高质量语料并开发评估基准。此外，企业需设置责任承担者以应对法律与舆论问责，并由模型解读员负责故障溯源与行为分析。

**Your score:** `[ 2]`

---

## 37. `ret_123_voice_cloning::https://deepmind.google/models/synthid/`

**Query:** 语音克隆与合成技术

**Title:** [AI] SynthID发布：AI生成内容的水印工具

**Summary:** Google DeepMind推出SynthID，一种用于标记和检测AI生成内容（包括图像、音频和文本）的数字水印工具。该技术通过在生成过程中嵌入不可见的水印来提高透明度和信任度，并可通过Gemini或独立的SynthID Detector进行验证。

**Your score:** `[ 0]`

---

## 38. `ret_124_red_teaming::https://github.com/darkrishabh/agent-skills-eval`

**Query:** AI 红队测试与安全评估

**Title:** [AI] agent-skills-eval：面向AI Agent技能的自动化评估框架

**Summary:** agent-skills-eval是一款开源CLI与SDK测试工具，用于量化评估AI Agent技能文件的实际效果。该框架通过对比加载与未加载技能上下文的目标模型输出，调用裁判模型进行自动化打分，并生成包含通过率与断言证据的静态HTML报告。工具原生支持OpenAI兼容API及自定义Provider，内置确定性工具调用断言检查，评估数据以JSON与JSONL格式留存，便于CI流水线集成与后续分析。项目基于TypeScript开发，采用MIT协议。

**Your score:** `[0 ]`

---

## 39. `ret_125_small_models::https://aws.amazon.com/blogs/machine-learning/customize-amazon-nova-models-with-amazon-bedrock-fine-tuning/`

**Query:** 小模型与模型蒸馏

**Title:** [AI] AWS推出Amazon Bedrock微调Amazon Nova模型指南

**Summary:** AWS更新了Amazon Bedrock服务，支持对Nova系列模型进行监督微调(SFT)、强化微调(RFT)与模型蒸馏。该方案通过自动化训练流程与参数高效微调技术，使开发者仅需将JSONL数据上传至对象存储即可完成定制化。实测在航空意图分类任务中准确率提升至97%，同时有效降低推理延迟与Token成本，并支持按调用计费，大幅降低了企业级领域模型定制的生产门槛。

**Your score:** `[ 0]`

---

## 40. `ret_127_microsoft_ai_infra::https://techcrunch.com/2026/04/27/openai-ends-microsoft-legal-peril-over-its-50b-amazon-deal/`

**Query:** 微软在 AI 基础设施的投入

**Title:** [商业] OpenAI与微软重订合作协议，化解500亿美元亚马逊交易法律风险

**Summary:** 微软与OpenAI宣布重新协商合作协议，终止微软对OpenAI产品及知识产权的独家授权，改为截至2032年的非独家许可。该条款解除OpenAI此前与亚马逊高达500亿美元投资协议引发的法律纠纷。新协议确立微软至2032年仍为OpenAI主要云合作伙伴，产品优先在Azure上线。财务条款方面，微软停止向OpenAI支付收入分成，而OpenAI向微软支付分成的条款延续至2030年并增设上限。微软目前仍持有OpenAI约27%的股份。

**Your score:** `[ 0]`

---

## 41. `ret_128_intel_foundry::https://developer.nvidia.com/blog/introducing-nvidia-fleet-intelligence-for-real-time-gpu-fleet-visibility-and-optimization/`

**Query:** Intel 代工与芯片制造

**Title:** [硬件] NVIDIA推出Fleet Intelligence服务实现GPU集群实时监控与优化

**Summary:** NVIDIA正式推出Fleet Intelligence托管服务，旨在为数据中心GPU/CPU集群提供实时监控与优化能力。该服务基于开源Agent，整合GPUd、DCGM及Attestation SDK技术，聚焦功耗、温度、性能、健康状态与安全验证，支持Blackwell、Hopper及Vera Rubin架构。通过可视化面板、实时告警与完整性校验，帮助企业提升算力资源利用率与运维可靠性。

**Your score:** `[ 0]`

---

## 42. `ret_128_intel_foundry::https://www.intel.com/content/www/us/en/content-details/850997/intel-assured-supply-chain-product-brief.html`

**Query:** Intel 代工与芯片制造

**Title:** [硬件] 英特尔可信供应链产品简报

**Summary:** 英特尔发布可信供应链（ASC）产品简报，旨在应对半导体供应链中的假冒组件与固件攻击风险。该方案通过数字化可验证的监管链技术，为数据中心、服务器及PC处理器提供完整的来源证明与安全保障。简报详细阐述了安全制造走廊与数字认证特性，面向IT企业及政府客户，以提升硅制造流程的透明度并支持行业安全标准。

**Your score:** `[ 1]`

---

## 43. `ret_130_multimodal_models::https://aws.amazon.com/blogs/machine-learning/nvidia-nemotron-3-nano-omni-model-now-available-on-amazon-sagemaker-jumpstart/`

**Query:** 多模态大模型进展

**Title:** [AI] 英伟达Nemotron 3 Nano Omni多模态模型现已在Amazon SageMaker JumpStart上线

**Summary:** AWS宣布NVIDIA Nemotron 3 Nano Omni多模态模型现已支持在SageMaker JumpStart上一键部署。该模型基于Mamba2混合专家架构，整合视觉、语音与语言编码器，支持131K上下文及FP8精度。企业开发者可通过控制台或Python SDK快速创建推理终端节点，实现视频、音频与文本的统一单次推理。实测表明其吞吐量较同类开源模型提升最高9倍，大幅优化了多模态智能体应用的延迟与架构复杂度。

**Your score:** `[ 2]`

---

## 44. `ret_13_ai_chip_supply::https://techcrunch.com/2026/05/04/nicolas-sauvage-is-betting-on-the-boring-parts-of-ai/`

**Query:** AI 芯片市场与供应链

**Title:** [硬件] Nicolas Sauvage押注AI底层硬件与物理AI技术

**Summary:** TDK Ventures管理4支基金总规模达5亿美元，投资主线聚焦AI底层硬件与物理基础设施。该基金于2020年投资推理芯片企业Groq，后者最新估值已升至69亿美元，投资组合同时涵盖固态变压器与钠离子电池。当前关注点转向专用仓储与巡检机器人，以及负责AI任务编排的CPU架构。同时密切追踪中国厂商利用AI压缩硬件原型迭代周期的趋势，指出突破物理灵巧性瓶颈将成为下一代制造业的核心竞争力。

**Your score:** `[ 2]`

---

## 45. `ret_13_ai_chip_supply::https://techcrunch.com/2026/05/06/ai-boom-pushes-samsung-to-1t/`

**Query:** AI 芯片市场与供应链

**Title:** [商业] 人工智能需求推动三星市值突破1万亿美元

**Summary:** 受人工智能需求推动，三星电子市值周三突破1万亿美元，股价单日涨幅超百分之十。公司近期财报显示利润同比增长八倍，核心驱动力为AI系统必需的高带宽内存芯片。目前三星正与英特尔就为苹果在美国本土代工芯片进行谈判，若达成协议将重塑全球半导体供应链。行业面临产能短缺，三星、SK海力士与美光均将资源向高利润内存产品倾斜。此外，三星内部员工计划罢工要求提高分红比例，其终端产品部门亦承受芯片成本上涨的压力。

**Your score:** `[ 2]`

---

## 46. `ret_17_ai_funding::https://techcrunch.com/2026/05/09/nvidia-has-already-committed-40b-to-equity-ai-deals-this-year/`

**Query:** AI 创业公司融资

**Title:** [商业] Nvidia年初已在AI领域投入超400亿美元股权投资

**Summary:** 据CNBC报道，Nvidia在2026年初已向AI企业承诺超400亿美元股权投资。其中最大单笔投资为向OpenAI注资300亿美元，此外还包括向玻璃制造商Corning投资32亿美元及数据中心运营商IREN投资21亿美元。数据显示，Nvidia今年已参与约24轮私营初创公司融资。尽管外界批评此类投资属于资金在供需双方间循环的循环交易，但分析师指出此举有助于构建竞争优势护城河。

**Your score:** `[2 ]`

---

## 47. `ret_18_enterprise_adoption::https://techcrunch.com/2026/04/22/the-most-interesting-startups-showcased-at-google-cloud-next-2026/`

**Query:** AI 在企业市场的落地

**Title:** [商业] Google Cloud Next 2026重点展示的AI初创企业

**Summary:** Google Cloud Next 2026期间，谷歌宣布设立7.5亿美元专项预算，支持合作伙伴向企业客户部署AI Agent，资金覆盖概念验证、云额度及部署返利。会议集中展示了依托谷歌云生态扩展业务的初创企业：Lovable推出企业级编码Agent，ARR达4亿美元；Notion（估值约110亿美元）与Gamma（估值21亿美元）分别集成Gemini模型与Nano Banana 2图像生成技术；vLLM团队创办的Inferact通过谷歌云获取Nvidia算力。另有ComfyUI、Vapi等十余家公司在医疗、零售及开发者工具领域加速接入。

**Your score:** `[2 ]`

---

## 48. `ret_20_multimodal::https://mistral.ai/news/vibe-remote-agents-mistral-medium-3-5`

**Query:** 多模态模型进展

**Title:** [AI] Mistral发布Medium 3.5模型及云端远程编程智能体

**Summary:** Mistral发布128B稠密模型Medium 3.5，支持256k上下文窗口，SWE-Bench Verified得分77.6%，超越Devstral 2及Qwen3.5 397B A17B。该模型支持云端异步编程代理，可并行运行并集成GitHub、Jira等工具。Le Chat新增工作模式，通过多步骤任务处理与工具调用完成复杂工作。API定价为输入1.5美元/百万token，输出7.5美元。开源权重采用修改版MIT协议，已在Hugging Face上线。

**Your score:** `[ 0]`

---

## 49. `ret_22_anthropic_customers::https://techcrunch.com/2026/05/04/anthropic-and-openai-are-both-launching-joint-ventures-for-enterprise-ai-services/`

**Query:** Anthropic enterprise customers

**Title:** [商业] Anthropic与OpenAI相继成立合资企业，布局企业级AI服务

**Summary:** Anthropic宣布联合黑石、高盛等成立合资企业，估值15亿美元，各方各出资3亿美元。OpenAI亦同步筹备类似合资公司，计划融资40亿美元，估值100亿美元。两家机构均旨在通过引入另类资产管理资金，拓展企业级AI服务渠道并采用前置部署工程师模式。目前，OpenAI与Anthropic的母公司估值分别高达8520亿美元与9000亿美元，正加速推进IPO进程。

**Your score:** `[2 ]`

---

## 50. `ret_24_opensource_releases::https://github.com/rust-lang/rust-forge/pull/1040`

**Query:** open source LLM releases

**Title:** [开发] Rust编译器仓库拟引入LLM使用规范

**Summary:** 核心议题：Rust语言核心仓库拟引入大语言模型使用规范。主要规则：禁止直接提交LLM生成的代码、文档或评论；允许私下使用LLM辅助调试、代码审查及非核心逻辑编写，但需明确标注并保证人工验证。政策目标：遏制低质量灌水提交，保留技术探索空间。当前提案正就披露机制与治理流程进行多团队联合评审。

**Your score:** `[ 0]`

---

## 51. `ret_25_chip_shortage::https://techcrunch.com/2026/05/06/five-architects-of-the-ai-economy-explain-where-the-wheels-are-coming-off/`

**Query:** AI chip shortage supply chain

**Title:** [AI] AI产业五大领军人物剖析算力瓶颈、能源限制与架构演进

**Summary:** 五位AI产业核心高管在Milken会议指出行业面临物理瓶颈。ASML预测芯片供应受限将持续2至5年。Google Cloud上季度营收超200亿美元，积压待交付收入达4600亿美元，正探索轨道数据中心以突破能源限制。Perplexity推出企业级数字代理，强调细粒度权限管控。Logical Intelligence发布2亿参数能量模型，宣称推理速度较主流大模型快数千倍且无需从头训练。物理AI部署面临地缘主权与先进制程芯片获取的双重制约。

**Your score:** `[ 1]`

---

## 52. `ret_32_xai_datacenter::https://techcrunch.com/2026/04/28/google-expands-pentagons-access-to-its-ai-after-anthropics-refusal/`

**Query:** xAI 数据中心 燃气轮机 诉讼

**Title:** [商业] 谷歌扩大五角大楼AI访问权限，此前Anthropic因拒绝合作遭起诉

**Summary:** 谷歌已向美国国防部开放其AI在机密网络中的访问权限。此前Anthropic因拒绝移除反监控与反自主武器限制，被五角大楼列为“供应链风险”并陷入诉讼，OpenAI与xAI已先行达成同类协议。谷歌合同虽包含限制争议性军事用途的条款，但实际法律约束力尚不明确。面对内部950名员工联名反对，谷歌仍持续推进该国防级AI商业化部署。

**Your score:** `[0 ]`

---

## 53. `ret_39_deepseek_v4::https://news.future-shock.ai/the-weight-of-remembering/`

**Query:** DeepSeek V4 模型发布

**Title:** [AI] 从300KB到69KB每令牌：LLM架构如何解决KV缓存问题

**Summary:** 文章分析了大语言模型KV缓存的演进：GPT-2每令牌占用300 KiB，Llama 3通过分组查询注意力降至128 KiB，DeepSeek V3采用多头潜在注意力压缩至68.6 KiB，Gemma 3引入滑动窗口实现选择性记忆。另有状态空间模型如Mamba完全摒弃KV缓存。这些架构变迁反映了对“记忆”本质的工程哲学思考。

**Your score:** `[ 0]`

---

## 54. `ret_40_claude_aws::https://aws.amazon.com/blogs/machine-learning/control-where-your-ai-agents-can-browse-with-chrome-enterprise-policies-on-amazon-bedrock-agentcore/`

**Query:** Claude AWS 平台

**Title:** [AI] AWS Bedrock AgentCore支持Chrome企业策略以管控AI代理浏览行为

**Summary:** AWS Bedrock AgentCore浏览器新增Chrome企业策略与自定义根证书支持。企业可通过S3下发JSON策略，在浏览器层强制管控AI代理的URL访问范围、密码管理及下载权限，实现安全策略与业务代码解耦。结合Secrets Manager托管的证书，代理可安全连接内网服务或企业代理。该架构通过控制与数据平面API集成，提供会话录制与审计能力，强化了AI代理在企业生产环境中的合规性与网络边界控制。

**Your score:** `[0 ]`

---

## 55. `ret_41_capex_2026::https://techcrunch.com/2026/04/22/the-most-interesting-startups-showcased-at-google-cloud-next-2026/`

**Query:** 四大科技巨头 2026 资本支出

**Title:** [商业] Google Cloud Next 2026重点展示的AI初创企业

**Summary:** Google Cloud Next 2026期间，谷歌宣布设立7.5亿美元专项预算，支持合作伙伴向企业客户部署AI Agent，资金覆盖概念验证、云额度及部署返利。会议集中展示了依托谷歌云生态扩展业务的初创企业：Lovable推出企业级编码Agent，ARR达4亿美元；Notion（估值约110亿美元）与Gamma（估值21亿美元）分别集成Gemini模型与Nano Banana 2图像生成技术；vLLM团队创办的Inferact通过谷歌云获取Nvidia算力。另有ComfyUI、Vapi等十余家公司在医疗、零售及开发者工具领域加速接入。

**Your score:** `[ 2]`

---

## 56. `ret_42_ai_wealth::https://news.ycombinator.com/item?id=47857461`

**Query:** AI 行业财富分化与从业者

**Title:** [生态] HN热议：对AI泛滥的疲惫与技术反思

**Summary:** 核心议题：生成式AI泛滥引发的公众疲劳与行业反思。主要争议：AI导致的内容同质化与职场替代风险，对比其作为开发辅助工具的实际效能。典型观点：用户普遍担忧AI削弱人类专业技能习得与创造力，并警惕无限AI监控对个体自由的侵蚀；部分开发者则肯定其在代码生成、系统学习中的初级助理价值。讨论整体反映对技术过度炒作与应用失当的批判性审视。

**Your score:** `[2 ]`

---

## 57. `ret_42_ai_wealth::https://techcrunch.com/2026/05/16/the-haves-and-have-nots-of-the-ai-gold-rush/`

**Query:** AI 行业财富分化与从业者

**Title:** [生态] AI淘金热中的财富分化与从业者职业焦虑

**Summary:** Menlo Ventures合伙人Deedy Das发文指出，当前AI热潮加剧科技行业贫富分化。据估算，OpenAI、Anthropic及Nvidia等公司约1万名核心人员已实现超2000万美元财富积累。与此同时，科技圈裁员潮持续，大量软件工程师面临技能贬值担忧。该现象引发行业对职业路径的广泛讨论，外界观点分化，指出当前AI技术同时具备创造财富与替代岗位的双重属性，加剧从业者对行业未来的不确定性。

**Your score:** `[ 2]`

---

## 58. `ret_43_ai_jobs::https://blogs.microsoft.com/blog/2026/05/05/how-frontier-firms-are-rebuilding-the-operating-model-for-the-age-of-ai/`

**Query:** AI 对就业市场的影响

**Title:** [AI] 前沿企业如何利用AI重构运营模式及Copilot更新

**Summary:** 微软发布2026年工作趋势指数，基于万亿级生产力数据与两万名员工调研，提炼出人机协作的四种模式。数据显示，49%的Copilot对话支持认知工作，80%的前沿专业用户已产出以往无法完成的工作。组织文化与经理支持对AI成效的影响是个人因素的2倍以上。同期微软推出Copilot Cowork移动端及插件生态，支持跨应用任务委派与Agent 365统一治理，助力企业构建AI运营模式。

**Your score:** `[ 2]`

---

## 59. `ret_43_ai_jobs::https://techcrunch.com/2026/05/04/as-workers-worry-about-ai-nvidias-jensen-huang-says-ai-is-creating-an-enormous-number-of-jobs/`

**Query:** AI 对就业市场的影响

**Title:** [AI] 英伟达CEO黄仁勋称AI正创造大量就业岗位并反驳失业担忧

**Summary:** 英伟达CEO黄仁勋在公开活动中指出，AI技术旨在自动化具体任务而非替代完整岗位，将助力美国实现再工业化并创造大量就业。其明确反对AI将引发大规模失业的悲观论调，认为过度炒作恐惧会阻碍技术普及。尽管企业界对AI就业带动效应持乐观预期，多家金融与学术机构仍预测受AI技术渗透影响，美国未来几年约15%的现有岗位面临淘汰，宏观经济的长期结构性影响尚待验证。

**Your score:** `[ 2]`

---

## 60. `ret_43_ai_jobs::https://www.citadelsecurities.com/news-and-insights/2026-global-intelligence-crisis/`

**Query:** AI 对就业市场的影响

**Title:** [AI] 软件工程师岗位需求攀升与AI就业市场分析

**Summary:** 美国软件工程师岗位发布量同比上涨11%，AI资本支出占GDP比重达2%。圣路易斯联储数据显示，生成式AI工作使用率保持稳定，未见指数级扩散迹象。技术扩散受算力成本与组织整合限制呈现S型曲线，自动化替代存在明确经济边界。宏观分析指出，AI属正向供给冲击，数据中心建设已带动招聘。综合历史规律，AI更可能作为劳动力补充以抵消增长阻力，而非引发大规模失业。

**Your score:** `[ 2]`

---

## 61. `ret_43_ai_jobs::https://www.wired.com/story/meta-layoffs-bad-vibes-mark-zuckerberg-ai/`

**Query:** AI 对就业市场的影响

**Title:** [商业] Meta利润创新高但员工士气低迷

**Summary:** Meta计划于5月20日裁员约10%。尽管公司实现创纪录利润，内部员工士气却处于历史低点，除高管外普遍感到不满。受此影响，位于爱尔兰的700多名Meta AI外包训练员也面临失业风险。此次人事调整旨在优化成本结构，但引发了关于企业治理与职场信任的讨论。

**Your score:** `[ 2]`

---

## 62. `ret_45_ai_video::https://blog.google/products-and-platforms/platforms/google-tv/enjoy-new-ways-to-create-search-and-stream-on-google-tv/`

**Query:** AI 视频生成技术

**Title:** [AI] Google TV集成Gemini AI模型推出图像与视频生成功能

**Summary:** Google TV宣布集成Gemini生态AI工具Nano Banana与Veo，支持电视端通过语音指令生成趣味图片与定制视频，首批面向美国TCL设备开放。系统新增语音检索Google Photos库功能，支持按语义定位特定照片并提供AI风格重绘。动态幻灯片功能适用于2GB及以上RAM设备。主页将于夏季新增“Short videos”推荐流，初期整合YouTube Shorts内容。

**Your score:** `[ 2]`

---

## 63. `ret_45_ai_video::https://techcrunch.com/2026/05/05/marc-lore-says-that-ai-will-soon-enable-anyone-open-a-restaurant/`

**Query:** AI 视频生成技术

**Title:** [AI] Marc Lore称AI将很快使任何人能够开设餐厅

**Summary:** Wonder推出“Wonder Create”平台，允许用户通过AI提示词在1分钟内自动生成餐厅品牌、菜单与定价。该虚拟品牌将接入其自动化厨房网络，现有120家门店明年预计扩至400家。单店配备12名员工及烹饪机器人，产能达700万份，目标2035年于2500平方英尺内运营1000个独立品牌。公司整合配送与食材供应链，试图通过AI与自动化技术重塑餐饮创业模式，但规模化可行性仍待验证。

**Your score:** `[ 0]`

---

## 64. `ret_47_gpt55_benchmark::https://openrouter.ai/announcements/gpt55-cost-analysis`

**Query:** GPT-5.5 benchmark performance

**Title:** [AI] GPT-5.5调价实测：实际使用成本与提示词长度分析

**Summary:** OpenRouter实测数据显示，GPT-5.5基础定价较5.4版本翻倍，输入输出单价分别达5美元与30美元/百万Token。分析表明，该模型对万字以上长提示词生成量减少19%至34%，短提示词下生成量反升。受此影响，用户实际综合成本增加49%至92%，不同工作流的成本效益呈现显著分化。

**Your score:** `[2 ]`

---

## 65. `ret_47_gpt55_benchmark::https://thinkpol.ca/2026/04/30/an-open-weights-chinese-model-just-beat-claude-gpt-5-5-and-gemini-in-a-programming-challenge/`

**Query:** GPT-5.5 benchmark performance

**Title:** [AI] 中国开源模型Kimi K2.6在编程挑战中击败多家西方前沿模型

**Summary:** 在近期AI编程挑战赛“Word Gem Puzzle”中，中国Moonshot AI的开源模型Kimi K2.6以22积分、7胜1负的成绩夺冠，小米MiMo V2-Pro位列第二。OpenAI的GPT-5.5与Anthropic的Claude Opus 4.7分列第三、第五。Kimi凭借动态滑动策略在大规模网格中表现优异。目前Kimi K2.6在AI智能指数中得分为54，与GPT-5.5（60）及Claude（57）差距缩小，表明头部开源模型已逼近闭源前沿水平。

**Your score:** `[ 0]`

---

## 66. `ret_51_google_products::https://blog.google/innovation-and-ai/models-and-research/google-deepmind/accelerator-ai-for-the-planet/`

**Query:** Google AI 最近的产品发布

**Title:** [商业] Google DeepMind在亚太地区推出AI环境风险加速器计划

**Summary:** Google DeepMind在亚太地区正式启动首届“AI for the Planet”加速器计划。该项目为期三个月，面向初创企业、科研团队与非营利组织开放，聚焦自然生态、气候预测、农业优化及能源管理等核心领域。入选机构将获得Google AI专家团队的定向指导与技术资源，支持其将前沿AI模型整合至现有项目中。计划首期线下训练营将于新加坡举办，旨在推动环境领域AI解决方案的规模化落地。

**Your score:** `[ 0]`

---

## 67. `ret_51_google_products::https://googlebook.google/`

**Query:** Google AI 最近的产品发布

**Title:** [硬件] 谷歌宣布推出专为Gemini设计的AI笔记本“Googlebook”

**Summary:** 谷歌计划于2026年秋季推出首款深度集成Gemini AI的笔记本电脑。核心功能包括基于大模型的Magic Pointer智能交互与自定义组件生成，并强化与Android 17及以上设备的生态联动，支持免安装跨端运行手机应用及直接读取手机文件。设备采用轻量化设计，目前官网仅开放订阅通知入口，详细芯片与续航规格尚未披露。

**Your score:** `[ 2]`

---

## 68. `ret_53_ai_agent_progress::https://github.com/facebookresearch/hyperagents`

**Query:** AI agent 智能体最新进展

**Title:** [AI] HyperAgents：自指式自我改进智能体

**Summary:** Meta发布HyperAgents框架，支持智能体通过自指机制优化任意可计算任务。项目包含任务代理与元代理核心模块，依赖多模态API（OpenAI/Anthropic/Gemini）。实验日志以分卷压缩包形式存储，附带安全警告提示执行模型生成代码存在潜在风险。

**Your score:** `[ 2]`

---

## 69. `ret_57_meta_ai_strategy::https://techcrunch.com/2026/05/05/meta-will-use-ai-to-analyze-height-and-bone-structure-to-identify-if-users-are-underage/`

**Query:** Meta 的 AI 战略与产品

**Title:** [AI] Meta将启用AI分析身高与骨骼结构以识别未成年用户

**Summary:** Meta宣布部署AI系统，通过分析图像与视频中的身高、骨骼结构等视觉特征，结合文本与互动上下文，识别Facebook及Instagram上未满13岁用户。该系统明确排除人脸识别，目前已在部分国家试点，疑似未成年账户将被停用直至通过官方年龄验证。此举措源于新墨西哥州就儿童安全问题判处Meta的3.75亿美元民事罚款。同时，Meta将包含默认隐私设置与消息限制的严格青少年账户模式扩展至欧盟27国与巴西，并于6月登陆英美地区。

**Your score:** `[2 ]`

---

## 70. `ret_57_meta_ai_strategy::https://www.theverge.com/tech/929091/meta-ai-threads-account-block`

**Query:** Meta 的 AI 战略与产品

**Title:** [AI] Meta在Threads测试AI账号功能但暂不支持屏蔽

**Summary:** Meta在Threads平台灰度测试AI账号标签功能，允许用户通过提及该AI获取问答与上下文信息。首批测试覆盖阿根廷、墨西哥、新加坡等五国。功能上线后用户发现官方未提供屏蔽入口，操作屏蔽时直接触发系统错误。该限制引发大量用户投诉与负面反馈。Meta近期持续加大AI研发投入，旨在通过AI集成提升社交平台用户活跃度。

**Your score:** `[2 ]`

---

## 71. `ret_61_startup_funding::https://techcrunch.com/2026/04/14/ai-datacenter-startup-fluidstack-in-talks-for-1b-round-at-18b-valuation-months-after-hitting-7-5b-says-report/`

**Query:** AI 初创公司融资

**Title:** [商业] AI数据中心初创公司Fluidstack正洽谈10亿美元融资，估值达180亿美元

**Summary:** AI数据中心初创公司Fluidstack正洽谈10亿美元融资，估值达180亿美元，较数月前75亿美元估值实现翻倍，本轮或由Jane Street领投。此前该公司已获Anthropic价值500亿美元的定制数据中心订单，业务涵盖得克萨斯州与纽约州项目。为聚焦美国市场，Fluidstack已将总部迁至纽约，并退出法国百亿欧元AI项目。目前其客户还包括Meta、Poolside及Mistral等。

**Your score:** `[0 ]`

---

## 72. `ret_61_startup_funding::https://techcrunch.com/2026/04/17/sources-cursor-in-talks-to-raise-2b-at-50b-valuation-as-enterprise-growth-surges/`

**Query:** AI 初创公司融资

**Title:** [商业] Cursor拟融资超20亿美元，估值达500亿美元，企业级业务增长强劲

**Summary:** AI编程初创公司Cursor正洽谈至少20亿美元融资，投前估值达500亿美元，由Thrive与a16z领投。公司预计2026年ARR将突破60亿美元，较同年2月的20亿美元实现三倍增长。依托自研Composer模型与低成本模型调用策略，Cursor整体毛利率已转正，其中企业级业务率先实现正向毛利，以降低对外部大模型供应商的依赖。

**Your score:** `[ 2]`

---

## 73. `ret_61_startup_funding::https://www.cnbc.com/2026/03/31/openai-funding-round-ipo.html`

**Query:** AI 初创公司融资

**Title:** [商业] OpenAI完成1220亿美元融资，估值达8520亿美元

**Summary:** OpenAI宣布完成1220亿美元融资，投后估值达8520亿美元，较此前公布的1100亿美元有所上调。公司目前月收入20亿美元，去年总收入131亿美元，尚未盈利。软银、Andreessen Horowitz等领投，个人投资者贡献30亿美元。

**Your score:** `[ 0]`

---

## 74. `ret_62_intel_ai_chip::https://morethanmoore.substack.com/p/an-interview-with-pat-gelsinger-2026`

**Query:** Intel 的 AI 芯片进展

**Title:** [硬件] 专访前Intel CEO Pat Gelsinger：硬科技投资与下一代算力架构展望

**Summary:** 前Intel CEO Pat Gelsinger目前任职于Playground Global，专注于硬科技投资。他指出AI推理性能需实现10000倍提升，未来算力将走向经典、AI与量子计算的“三位一体”异构架构。其投资布局涵盖硅光互连、超导逻辑及非冯·诺依曼计算设备。在半导体制造方面，他押注自由电子激光等下一代光源技术以突破光刻瓶颈，并强调扩建核能是解决AI算力能耗危机的核心路径。

**Your score:** `[2 ]`

---

## 75. `ret_62_intel_ai_chip::https://techcrunch.com/2026/04/01/cognichip-wants-ai-to-design-the-chips-that-power-ai-and-just-raised-60m-to-try/`

**Query:** Intel 的 AI 芯片进展

**Title:** [商业] Cognichip获6000万美元融资，拟用AI加速芯片设计

**Summary:** Cognichip宣布完成6000万美元融资，由Seligman Ventures领投，Intel CEO Lip-Bu Tan通过Walden Catalyst Ventures参投并加入董事会。公司称其专用AI模型可降低75%以上芯片开发成本、缩短超一半周期，已累计融资9300万美元。目前尚未公布客户或量产案例。

**Your score:** `[ 0]`

---

## 76. `ret_63_sora_like_video::https://blog.google/products/ads-commerce/demand-gen-drop-march-2026/`

**Query:** AI 视频生成模型与产品

**Title:** [AI] 利用AI视频生成与创作者合作提升广告效果

**Summary:** Google Ads推出Veo工具，可从静态图生成高质量视频变体以提升广告强度。YouTube创作者合作方案在Shorts上平均带来30%转化提升。Demand Gen现支持优化关注后续观看，以增加频道观看时长。

**Your score:** `[ 2]`

---

## 77. `ret_64_ai_coding_tools::https://blog.orhun.dev/code-responsibly/`

**Query:** AI 编程助手与 coding 工具

**Title:** [开发] 编写更少代码，承担更多责任：AI辅助编程的实践思考

**Summary:** 本文探讨AI辅助编程对开发者工作流的影响。作者结合使用Copilot与Codex的经验，提出采用混合策略：利用AI处理重复性任务，核心逻辑由开发者手动编写并逐行审查，以兼顾效率与代码质量。文章指出海量AI生成项目可能引发可维护性下降与开源许可争议，强调开发者仍需对最终产品负责，AI应作为提升效率的工具而非替代工程严谨性。

**Your score:** `[ 1]`

---

## 78. `ret_64_ai_coding_tools::https://twill.ai`

**Query:** AI 编程助手与 coding 工具

**Title:** [AI] Twill：基于云端代理的自动化代码交付平台

**Summary:** Twill.ai（YC S25）推出云端编程代理平台，支持并行调度Claude Code、OpenCode及Codex等模型。平台采用标准化工作流与隔离沙盒环境，自动完成代码构建、测试验证与PR生成。系统深度集成GitHub、Linear与Slack，并提供错误修复、依赖更新、测试维护等预设自动化模板。该工具旨在减少开发者上下文切换，实现工程任务的全天候异步处理。

**Your score:** `[2 ]`

---

## 79. `ret_65_robotics_embodied::https://deepmind.google/blog/gemini-robotics-er-1-6/`

**Query:** 机器人与具身智能

**Title:** [AI] Google DeepMind发布Gemini Robotics-ER 1.6，强化具身推理与机器人任务能力

**Summary:** Google DeepMind正式发布Gemini Robotics-ER 1.6模型，专注提升机器人具身推理能力。该版本强化了空间推理、多视角理解与任务成功检测，并新增仪表读取功能，可精准解析压力表与液位计，相关能力经与Boston Dynamics合作验证。模型采用代理视觉与代码执行结合架构，通过放大图像与比例估算实现高精度读数。相比前代及Gemini 3.0 Flash，其在指针定位、计数及安全合规（文本风险识别提升6%，视频提升10%）方面表现更优，目前已通过Gemini API与Google AI Studio向开发者开放。

**Your score:** `[2 ]`

---

## 80. `ret_65_robotics_embodied::https://techcrunch.com/2026/05/01/meta-buys-robotics-startup-to-bolster-its-humanoid-ai-ambitions/`

**Query:** 机器人与具身智能

**Title:** [商业] Meta收购人形机器人初创公司以强化具身智能布局

**Summary:** Meta以未披露金额收购人形机器人初创公司ARI，研发团队将并入Meta超级智能实验室。该公司专注开发执行物理任务的人形机器人基础模型。联合创始人Xiaolong Wang与Lerrel Pinto分别具备英伟达研究员及高校教职背景。此次并购旨在强化Meta在全身机器人控制与自主学习的模型设计能力，顺应AI通过物理交互训练迈向通用人工智能的趋势。

**Your score:** `[2 ]`

---

## 81. `ret_65_robotics_embodied::https://www.firgelli.com/pages/humanoid-robot-actuators`

**Query:** 机器人与具身智能

**Title:** [硬件] 人形机器人执行器：完整工程指南

**Summary:** 本文系统解析人形机器人执行器设计原则。单步承受两至三倍体重冲击，要求执行器具备高反驱性。核心指标为比扭矩与比力。主流架构采用应变波齿轮处理精密旋转，行星滚柱丝杠承担高负载。控制依赖场定向与模型预测算法。热管理是核心瓶颈，液冷可将持续扭矩提至峰值六成。定制关节需满足七十公斤级机器人膝盖一百五十牛米峰值扭矩、低于一牛米反驱力及小于二点五公斤质量等指标。

**Your score:** `[2 ]`

---

## 82. `ret_67_chatgpt_features::https://techcrunch.com/2026/04/30/openai-announces-new-advanced-security-for-chatgpt-accounts-including-a-partnership-with-yubico/`

**Query:** ChatGPT 的新功能

**Title:** [安全] OpenAI推出ChatGPT高级账户安全计划并联手Yubico发布安全密钥

**Summary:** OpenAI正式推出ChatGPT高级账户安全计划，并与Yubico合作发布联名款安全密钥。该方案面向政要、记者及企业用户，通过硬件加密标识抵御钓鱼攻击，旨在降低敏感数据未授权访问风险。作为OpenAI数字防御战略的一环，该功能提供强保护但附带不可恢复特性，密钥丢失将导致账户及数据永久无法找回。

**Your score:** `[ 2]`

---

## 83. `ret_67_chatgpt_features::https://techcrunch.com/2026/05/16/openai-co-founder-greg-brockman-reportedly-takes-charge-of-product-strategy/`

**Query:** ChatGPT 的新功能

**Title:** [商业] OpenAI联合创始人Greg Brockman正式接管产品战略

**Summary:** OpenAI联合创始人兼总裁Greg Brockman正式接管公司产品战略，接替处于医疗休假期的Fidji Simo。其计划将ChatGPT与编程产品Codex整合为单一体验，以聚焦智能体技术并覆盖消费与企业市场。该调整落实了公司重核主线的战略指令，目前已终止Sora及科学业务等支线项目，集中资源打造AI超级应用。

**Your score:** `[ 0]`

---

## 84. `ret_68_ai_security::https://aphyr.com/posts/417-the-future-of-everything-is-lies-i-guess-safety`

**Query:** AI 安全漏洞与攻击

**Title:** [安全] 大语言模型的安全隐患与对齐困境

**Summary:** 大语言模型的对齐机制难以彻底消除安全风险，未对齐模型易于被训练和部署。提示注入攻击使模型在处理不可信输入时面临数据泄露隐患，赋予模型外部操作权限存在严重安全缺陷。生成式AI技术降低了定向欺诈与自动化网络骚扰的实施成本，加剧社会信任体系压力。此外，自动化军事AI应用与内容审核工作带来物理及心理层面的安全挑战，相关防护机制亟待完善。

**Your score:** `[ 2]`

---

## 85. `ret_73_llama_updates::https://api-docs.deepseek.com/news/news260424`

**Query:** Meta Llama 模型更新

**Title:** [AI] DeepSeek发布V4预览版：推出Pro与Flash双模型，全面支持百万级上下文

**Summary:** DeepSeek正式发布并开源V4预览版，推出Pro与Flash两款模型。其中Pro版总参数1.6T、激活49B，Flash版总参数284B、激活13B。全系标配100万上下文窗口，并支持思考与非思考双模式。API已全面更新，兼容OpenAI与Anthropic接口格式。官方同时宣布，旧版chat与reasoner模型将于2026年7月24日正式退役。

**Your score:** `[ 0]`

---

## 86. `ret_74_ai_chip_competition::https://xn--gckvb8fzb.com/hold-on-to-your-hardware/`

**Query:** AI 芯片市场竞争

**Title:** [硬件] 硬件价格飙升与消费者选择萎缩的警示

**Summary:** 受数据中心与“AI”需求驱动，全球DRAM与NAND产能被企业客户锁定，消费级硬件面临长期短缺。三星与SK海力士形成内存双寡头，西部数据等厂商退出消费市场。PC、游戏机、树莓派等设备涨价或断货，行业转向订阅制与云端租赁模式，消费者数字主权面临侵蚀。

**Your score:** `[ 1]`

---

## 87. `ret_77_amd_accelerator::https://www.anuragk.com/blog/posts/Taalas.html`

**Query:** AMD 的 AI 加速器

**Title:** [硬件] Taalas将LLM“打印”到芯片上

**Summary:** Taalas推出专为Llama 3.1 8B设计的ASIC芯片，通过将模型权重直接固化在硅片中，实现17000 tokens/秒的推理速度，较GPU方案快10倍，能耗与拥有成本均降低90%。芯片采用固定功能架构，利用自研‘魔法乘法器’在单晶体管级完成4-bit计算，规避内存墙问题，并通过定制金属层快速适配新模型，仅需两个月流片周期。

**Your score:** `[ 0]`

---

## 88. `ret_78_opensource_ecosystem::https://aws.amazon.com/blogs/machine-learning/fine-tune-llm-with-databricks-unity-catalog-and-amazon-sagemaker-ai/`

**Query:** 开源大模型生态

**Title:** [AI] 结合Databricks Unity Catalog与Amazon SageMaker AI微调大语言模型

**Summary:** AWS发布大语言模型微调的工程实践指南，详细阐述了通过EMR Serverless进行数据预处理，并集成SageMaker AI与Unity Catalog以实现跨服务数据治理与血缘追踪的架构方案。该方案确保了在模型训练过程中访问控制与审计合规的无缝衔接，为受监管行业的生产级生成式AI工作负载提供了可落地的部署参考。

**Your score:** `[ 0]`

---

## 89. `ret_78_opensource_ecosystem::https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f`

**Query:** 开源大模型生态

**Title:** [AI] LLM Wiki模式：基于大模型的个人知识库构建范式

**Summary:** 本文提出一种基于大模型的个人知识库构建范式，旨在替代传统RAG检索机制。该模式通过大模型增量读取原始资料并持续维护结构化的Markdown维基，实现知识的持久积累与动态更新。架构包含原始数据层、大模型托管的维基层及规范配置层。操作流程涵盖资料摄取、交叉引用查询与定期健康检查。该方案将知识整理维护成本转移至大模型，适用于学术研究及企业文档管理等场景。

**Your score:** `[ 1]`

---

## 90. `ret_78_opensource_ecosystem::https://unix.foo/posts/local-ai-needs-to-be-norm/`

**Query:** 开源大模型生态

**Title:** [AI] 本地AI应成为软件开发常态

**Summary:** 文章指出当前软件过度依赖云端大模型API，导致系统脆弱且存在隐私风险。作者主张将本地AI作为标准开发范式，强调端侧芯片算力已满足摘要、分类等数据转换需求。以苹果FoundationModels API为例，本地模型可直接处理用户数据，避免第三方服务依赖，并通过定义结构化类型提升输出可靠性。该模式在保障数据隐私的同时，降低了分布式系统的维护成本。

**Your score:** `[1 ]`

---

## 91. `ret_78_opensource_ecosystem::https://www.mendral.com/blog/frontier-model-lower-costs`

**Query:** 开源大模型生态

**Title:** [AI] 采用分层代理架构降低大模型日志分析成本

**Summary:** 本文介绍一种面向高容量事件数据的分层大模型代理架构。系统采用轻量模型作为初级筛选器，拦截约百分之八十的重复故障，使前沿模型仅处理复杂任务。轻量模型承担百分之六十五的输入量但仅占百分之三十六成本，匹配成本较完整调查低二十五倍。前沿模型负责推理与子任务调度，通过结构化数据库接口按需检索上下文，避免全量注入。该方案使整体大模型支出减半，可复用于安全日志与物联网遥测等场景。

**Your score:** `[ 1]`

---

## 92. `ret_79_inference_speedup::https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/`

**Query:** 大模型推理加速

**Title:** [AI] Google为Gemma 4发布多词元预测草案器，推理速度最高提升3倍

**Summary:** 谷歌正式发布Gemma 4系列的多词元预测草案器，采用推测解码架构优化大模型生成流程。测试表明，该方案在完全保留原模型推理质量的前提下，最高实现3倍生成速度提升。草案器通过复用目标模型激活状态与KV缓存，显著降低了消费级显卡与边缘设备的内存带宽瓶颈。相关模型权重已按Apache 2.0协议开源，并适配vLLM、MLX、Hugging Face及Ollama等主流推理框架。

**Your score:** `[2 ]`

---

## 93. `ret_80_xai_grok::https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/`

**Query:** xAI 与 Grok 模型

**Title:** [开发] 在 Kubernetes 上部署解耦的 LLM 推理工作负载

**Summary:** 本文介绍在 Kubernetes 上部署解耦式大语言模型推理的方法，通过分离预填充与解码阶段实现独立扩缩容和更高 GPU 利用率，并探讨 Grove 与 KAI Scheduler 等工具对多角色协调调度的支持。

**Your score:** `[0 ]`

---

## 94. `ret_81_ai_voice::https://www.feldera.com/blog/ai-agents-arent-coworkers-embed-them-in-your-software`

**Query:** AI 语音合成与克隆

**Title:** [AI] 代理并非同事，应将其嵌入软件架构

**Summary:** 针对当前AI代理拟人化交互导致的高认知负荷问题，文章提出应将其作为平静技术嵌入软件底层。推荐设计模式包括：提供命令行接口以降低交互开销、采用声明式配置明确目标状态、引入协调循环实现系统自动收敛。结合变更数据捕获技术，系统可实时推送增量事件流，代理仅需针对变更响应而无需轮询，从而构建低噪音、高效率的自动化工作流。

**Your score:** `[ 0]`

---

## 95. `ret_83_ai_alignment::https://techcrunch.com/2026/03/01/openai-shares-more-details-about-its-agreement-with-the-pentagon/`

**Query:** AI 对齐与安全研究

**Title:** [AI] OpenAI公布与五角大楼协议详情

**Summary:** OpenAI与美国国防部达成协议，允许其模型在保密环境中部署。尽管CEO Sam Altman承认协议仓促且引发争议，但OpenAI强调不会用于大规模国内监控、自主武器系统或高风险自动化决策。公司通过多层保护措施确保安全，并限制部署到云端API以防止直接集成到武器系统中。Anthropic因未能达成类似协议而被标记为供应链风险。

**Your score:** `[2 ]`

---

## 96. `ret_86_quantum::https://www.nist.gov/news-events/news/2026/04/any-color-you-nist-scientists-create-any-wavelength-lasers-tiny-circuits`

**Query:** 量子计算进展

**Title:** [硬件] NIST研发全波长集成光子芯片

**Summary:** NIST团队于《Nature》发表集成光子芯片技术，通过三维堆叠五氧化二钽与铌酸锂等材料，在硅晶圆上实现任意波长激光生成。该技术将上万条光子电路集成于微小芯片中，单电路输出独立颜色，有效解决量子计算与光学原子钟传统激光器体积大、功耗高的痛点。NIST已联合Octave Photonics推进规模化量产，未来将拓展至AI算力互连、VR显示及便携导航领域。

**Your score:** `[2 ]`

---

## 97. `ret_95_ai_jobs::https://economist.com/finance-and-economics/2026/04/13/the-tech-jobs-bust-is-real-dont-blame-ai-yet`

**Query:** AI 对就业与裁员的影响

**Title:** [商业] 科技业裁员潮属实，暂非AI所致

**Summary:** 美国科技业正经历裁员潮，Oracle、Block、亚马逊与Meta等巨头接连宣布削减岗位。其中Block裁员超4000人，约占其员工半数。2022至2025年间，科技“七巨头”总雇佣规模几乎停滞。旧金山科技相关就业自2023年初以来下降3%。分析指出本轮人员精简主要受宏观经济与行业周期影响，当前阶段尚不能将责任归咎于人工智能。

**Your score:** `[2 ]`

---

## 98. `ret_96_enterprise_agent_platform::https://blogs.microsoft.com/blog/2026/04/21/accelerating-frontier-transformation-with-microsoft-partners/`

**Query:** 企业级 AI agent 平台

**Title:** [AI] 微软发布前沿转型框架与Agent 365套件，推动合作伙伴生态规模化落地

**Summary:** 微软发布“前沿转型”框架及Microsoft 365 E7与Agent 365套件，计划于2026年5月全面上市。该套件集成Copilot与Agent 365控制平面，旨在为AI Agent提供统一治理与安全底座。数据显示，超90%财富500强企业已采用Copilot，IDC预测2028年AI Agent数量将达13亿。微软同步更新合作伙伴计划与专家认证，依托Marketplace构建近3000亿美元生态商机，推动中小企业与渠道伙伴实现AI规模化落地。

**Your score:** `[2 ]`

---

## 99. `ret_98_earnings_ai::https://techcrunch.com/2026/05/07/kodiak-ai-raises-100m-at-a-steep-discount-sending-its-stock-tumbling-37/`

**Query:** 科技公司财报中的 AI 业务

**Title:** [商业] Kodiak AI折价融资1亿美元致股价暴跌37%

**Summary:** 自动驾驶卡车公司Kodiak AI以每股6.5美元折价发行股票融资1亿美元，致其盘后股价下跌37%。公司一季度营收达180万美元，同比微增，但运营亏损扩大至3780万美元。资金主要用于扩展干线及工业场景业务，目前已与多家物流及工业企业合作，并计划于2026年底前完成技术验证后推进无人驾驶部署。

**Your score:** `[ 0]`

---

## 100. `ret_98_earnings_ai::https://techcrunch.com/2026/05/08/airbnb-says-ai-now-writes-60-of-its-new-code/`

**Query:** 科技公司财报中的 AI 业务

**Title:** [商业] Airbnb称AI已编写60%新代码，Q1营收增长18%

**Summary:** Airbnb在2026年第一季度财报中披露，其AI工具已编写60%的新代码，并自动处理40%的客服咨询。财务数据显示，该季度营收同比增长18%至27亿美元，净利润增长3.9%至1.6亿美元，预订夜数增至1.562亿。公司CEO指出，尽管AI显著提升了工程杠杆，但当前聊天机器人界面在图文展示、直接操作及多用户协同方面仍存在短板，尚未完全契合旅游电商的交易场景。

**Your score:** `[0 ]`

---
