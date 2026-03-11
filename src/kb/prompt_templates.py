"""
海关业务场景化 Prompt 模板。

这里维护 40+ 个轻量级场景模板，不直接替代主 Prompt，而是在 RAG 生成前根据问题关键词
挑选最相关的 1~3 个模板作为“补充回答约束”，让模型更贴近海关业务表达方式与答题重点。
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class PromptScenario:
    """单个业务场景模板。"""

    name: str
    keywords: tuple[str, ...]
    guidance: str


SCENARIO_PROMPT_TEMPLATES: List[PromptScenario] = [
    PromptScenario("通关流程", ("通关", "流程", "步骤", "怎么办理"), "优先按流程顺序回答，明确申报、审单、查验、放行等关键环节。"),
    PromptScenario("申报材料", ("材料", "资料", "单证", "提交什么"), "优先列出所需材料，并说明是否需要补充证明、随附单据或前置资质。"),
    PromptScenario("报关单填制", ("报关单", "填报", "填制", "申报要素"), "优先说明字段填写要求、申报要素与常见漏填误填风险。"),
    PromptScenario("商品归类", ("归类", "税号", "商品编码", "hs"), "优先提示归类判断依据、关键属性与仅凭当前证据无法最终定类的边界。"),
    PromptScenario("税率税款", ("税率", "税款", "关税", "增值税", "消费税"), "优先说明税种、计税口径、税率适用条件；证据不足时不要猜测税率。"),
    PromptScenario("原产地规则", ("原产地", "产地证", "协定税率", "原产资格"), "优先说明原产地判定与优惠适用条件，明确是否需要原产地证明。"),
    PromptScenario("减免税", ("减免税", "免税", "征免"), "优先回答适用条件、审批材料、后续监管要求。"),
    PromptScenario("AEO认证", ("aeo", "高级认证", "海关认证"), "优先说明认证条件、申请流程、管理要求与可享便利措施。"),
    PromptScenario("跨境电商", ("跨境电商", "9610", "9710", "9810", "1210"), "优先区分业务模式、监管代码、适用主体与申报差异。"),
    PromptScenario("一般贸易", ("一般贸易",), "优先说明一般贸易下的常规单证、申报流程与税费要求。"),
    PromptScenario("加工贸易", ("加工贸易", "手册", "保税料件"), "优先说明手册管理、核销、边角料或残次品处置要求。"),
    PromptScenario("保税监管", ("保税", "保税区", "保税仓", "综保区"), "优先说明保税区域类型、账册/货物流转要求与监管边界。"),
    PromptScenario("暂时进出境", ("暂时进出境", "暂时进口", "复运出境"), "优先说明期限、担保、复运要求与不能逾期的风险。"),
    PromptScenario("转关运输", ("转关", "转关运输"), "优先说明转关条件、申报节点、途中监管与异常处理。"),
    PromptScenario("舱单运输工具", ("舱单", "运抵", "运输工具", "提运单"), "优先说明舱单、运抵、提运单数据之间的对应关系和时点要求。"),
    PromptScenario("查验处置", ("查验", "开箱", "异常"), "优先说明查验触发、配合要求、常见异常和后续处理。"),
    PromptScenario("放行提货", ("放行", "提货", "放行后"), "优先说明放行条件、提货衔接和后续监管注意事项。"),
    PromptScenario("企业备案", ("备案", "注册", "登记"), "优先说明备案主体、办理渠道、所需材料和生效条件。"),
    PromptScenario("企业资质许可", ("资质", "许可", "许可证", "审批"), "优先区分海关要求与行业主管部门前置许可。"),
    PromptScenario("行政处罚稽查", ("稽查", "处罚", "违规", "处罚标准"), "优先说明风险点、合规义务与是否需进一步咨询执法部门。"),
    PromptScenario("知识产权保护", ("知识产权", "侵权", "备案保护"), "优先说明备案、扣留、权利证明和侵权风险提示。"),
    PromptScenario("进出口食品", ("食品", "进口食品", "出口食品"), "优先说明准入、标签、检验检疫与合规证明要求。"),
    PromptScenario("化妆品", ("化妆品",), "优先说明备案注册、标签、成分与进口通关关注点。"),
    PromptScenario("医疗器械", ("医疗器械",), "优先说明注册备案、分类管理、单证要求与监管特殊性。"),
    PromptScenario("药品", ("药品", "医药"), "优先说明药品监管资质、准入、单证和高风险要求。"),
    PromptScenario("危险化学品", ("危险品", "危化品", "危险化学品"), "优先说明危险属性申报、包装标签、运输与查验风险。"),
    PromptScenario("动植物检疫", ("动植物", "检疫", "植物", "动物"), "优先说明准入名单、检疫审批、证书和疫情风险要求。"),
    PromptScenario("木质包装", ("木质包装", "ippc"), "优先说明木质包装检疫要求、标识和处置规则。"),
    PromptScenario("固废洋垃圾", ("固废", "洋垃圾", "废物"), "优先说明禁止/限制进口边界，证据不足时不要模糊放宽。"),
    PromptScenario("两用物项", ("两用物项", "出口管制"), "优先说明许可、用途审查与合规风险。"),
    PromptScenario("个人物品行邮", ("行邮", "个人物品", "邮递", "快件"), "优先区分个人物品与货物，说明税收和监管差异。"),
    PromptScenario("快件监管", ("快件", "国际快递"), "优先说明快件渠道监管特点、申报简化边界和资料要求。"),
    PromptScenario("旅客通关", ("旅客", "携带", "入境", "出境"), "优先说明旅客携带物品申报、免税额度和申报义务。"),
    PromptScenario("免税额度", ("免税额", "免税额度"), "优先回答额度、适用对象、超限后的处理方式。"),
    PromptScenario("出口退税衔接", ("出口退税",), "优先说明海关放行、单证衔接与税务环节的边界。"),
    PromptScenario("退运返修", ("退运", "返修", "退货"), "优先说明退运原因、流程、税费影响和单证要求。"),
    PromptScenario("样品广告品", ("样品", "广告品"), "优先说明是否征税、是否可简化申报及所需证明。"),
    PromptScenario("展会展品", ("展品", "展会", "参展"), "优先说明暂时进出境、担保、复运与处置要求。"),
    PromptScenario("邮政渠道", ("邮政", "ems"), "优先说明邮政渠道的申报、税费和监管限制。"),
    PromptScenario("电商零售进口", ("零售进口", "跨境零售"), "优先区分零售进口清单申报与一般贸易申报。"),
    PromptScenario("出口管制许可证", ("许可证", "出口许可"), "优先说明许可类型、申办节点与无证风险。"),
    PromptScenario("海关数据统计", ("数据", "统计", "进口量", "出口量", "名单"), "优先区分知识库事实回答与数据库查询结果，不得虚构统计值。"),
    PromptScenario("海关政策解读", ("政策", "公告", "解读", "通知"), "优先先给政策结论，再列适用对象、条件与执行口径。"),
    PromptScenario("税收担保", ("担保", "保证金"), "优先说明触发场景、担保形式和解除条件。"),
    PromptScenario("信用管理", ("信用", "失信", "信用等级"), "优先说明信用等级、管理措施与修复路径。"),
    PromptScenario("核查补税", ("补税", "追征", "核查"), "优先说明触发原因、处理流程、补救建议与时效风险。"),
]


def select_prompt_templates(question: str, max_templates: int = 3) -> List[PromptScenario]:
    """按关键词命中数选出最相关的场景模板，最多返回 max_templates 个。"""

    q = (question or "").lower().strip()
    if not q:
        return []
    scored: List[tuple[int, int, PromptScenario]] = []
    for idx, template in enumerate(SCENARIO_PROMPT_TEMPLATES):
        hits = sum(1 for keyword in template.keywords if keyword and keyword.lower() in q)
        if hits > 0:
            scored.append((hits, -idx, template))
    scored.sort(reverse=True)
    return [template for _, _, template in scored[:max_templates]]


def render_prompt_template_guidance(question: str, max_templates: int = 3) -> str:
    """将匹配到的场景模板渲染为 Prompt 可直接注入的补充说明。"""

    selected = select_prompt_templates(question, max_templates=max_templates)
    if not selected:
        return "未命中特定海关业务场景模板，请按通用海关政策咨询方式作答：先结论，后依据，结论必须有证据引用。"
    lines = ["本题命中的海关业务场景模板："]
    for item in selected:
        lines.append(f"- {item.name}：{item.guidance}")
    return "\n".join(lines)
