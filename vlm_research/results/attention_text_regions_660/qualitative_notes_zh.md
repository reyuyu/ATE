# 六组 COCO 原图与完整 Caption 人工核对

案例按 `qualitative_selection.json` 的固定规则事后选择：每个层段 Attention OFF 的最大 SPICE 损害及最大收益/最小损害各一张，已选图片不重复。不是随机代表样本，也没有改变任何总体统计分母。全部七条件原文、分数和词序差异保存在 `qualitative_captions.csv`，图为 `qualitative_six_coco.pdf`。下列核对只作描述性说明，不新增对象/动作准确率指标。

## 531134：红绿灯反射，明显退化

原图中心是带悲伤脸图案的交通灯，叠有街景反射。Full 保留了“sad face”和“reflection”，Late Text OFF 与 Full 相同。Early Attention OFF 丢失这两项，并把关系改成“on a street sign”；Middle Attention OFF 出现“traffic light is a yellow building”的对象混同。Late Attention OFF 转述背景街景，忽略了前景带脸交通灯。Early Text OFF 在开头后持续重复“a”，不是可用 Caption。

## 360772：绿色浴室，指标改善不等于更好描述

可见马桶、绿色脱落墙面和水箱上的蓝色液体瓶，Full 描述了马桶及瓶。Early Attention OFF 的 SPICE 从 0.0667 升到 0.2143，却输出多次“a bathroom with”，并漏掉马桶和瓶；这是评分改善但语言可用性恶化的反例。Middle Attention OFF 保留马桶和绿墙但重复“toilet”。Middle Text OFF 的“toilet brush”、Late Attention OFF 的“small window”在此图中不能可靠确认，不应当据分数当成新增正确细节。Early Text OFF 退化为“room”子词循环。

## 510095：持棒者与黄花，中层细节丢失

原图可见人、持握的棒状器具及大量黄色花。Full 说“holding a baseball bat in a field with yellow flowers”；Middle Attention OFF 只剩“standing in the grass”，丢失持棒和花的描述，SPICE 从 0.4800 降到 0.1053。Middle Text OFF 保留持棒但省略黄色花，并加入“ready to swing”的动作推测；Late Text OFF 与 Full 相同。Late Attention OFF 加入“getting ready to hit a ball”等未被单张图直接证明的意图。Early Attention OFF 虽重复持棒关系但语句无法正常结束；Early Text OFF 则退化为“a”循环。

## 381563：餐盘，内容概括与评分不一致

原图是多种食物置于白盘，可见米饭和蔬菜；肉类及“rice pudding”的精确食材类别不能仅凭图可靠判定。Middle Attention OFF 的 SPICE 从 Full 的 0.1667 升至 0.2703，但语句“a plate of food is a white plate ...”较生硬。Middle Text OFF 概括成“a plate of food on a table”，丢失食物细节，却仍有更高 SPICE。Early Attention OFF 同样出现 SPICE 提升，但连续重复“plate of food”，说明该指标不能替代语言流畅性核对。

## 320703：厨房，后层复读与对象遗漏

原图可见冰箱、灶台、洗涤区等厨房对象。Full 列出 refrigerator/stove/sink。Late Attention OFF 从第一个“A”之后进入“small”循环，SPICE 降到 0，未能描述主要对象。Middle Attention OFF 把整个厨房说成“white and black checkered tile”，丢失关键设备。Middle Text OFF 保留灶台但漏掉冰箱和水槽；Late Text OFF 保留冰箱和灶台。Early Attention OFF 出现不完整单词和重复“paper towelcome”，Early Text OFF 重复“with”。

## 415073：西装男子，细节增加的反例

图中可见穿西装、带红白相间领带的男子，面向相机且有微笑表情。Late Attention OFF 增加领带颜色条纹，SPICE 从 0.1714 升至 0.3590，是干预后部分可见细节更丰富的案例；“sitting at a table”并不能仅凭裁切图可靠确认，不把整句视为全真。Middle Text OFF 加入红色领带并保留微笑，Late Text OFF 与 Full 相同。Early 两条件均明显重复，不能因出现正确名词就判为成功。

## 解读边界

- 这些案例同时包含显著负例、评分改善但语言退化的反例，以及可见细节增加的案例；不用于估计总体出现频率。
- 全量 `coco_object_action_terms.csv` 的 NN*/VB* 词元增删是词汇描述差异，含助动词，不是人工视觉事实标签，也不证明对象/动作幻觉。
- Early Text OFF 全部 60 图无 EOS，均达 96 Token 上限；其问题不仅是对象/动作选择，更包括历史文本约束和终止行为。不能据此直接定位“语言中枢”。
