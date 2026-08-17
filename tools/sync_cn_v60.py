"""Sync CN version with EN v45 changes."""
import re

filepath = r"e:\pytorchFile\YSC_2\paper\TWC_CN_34.tex"
with open(filepath, "r", encoding="utf-8-sig") as f:
    content = f.read()

# Update version
content = content.replace(
    "%  版本:  59 --- 导师批注修订v22",
    "%  版本:  60 --- 导师批注修订v23：与EN v45同步 -- 全文语言润色，移除'动力学'，去除括号解释，去除连续冒号，平衡Related Work，改进子节标题"
)

# 1. Replace "动力学" (but not in LaTeX labels)
replacements_dynamics = [
    ("梯度驱动动力学", "梯度驱动集中机制"),
    ("离散动力学", "离散演化方程"),
    ("负载动力学", "负载演化"),
    ("卫星负载动力学", "卫星负载演化"),
    ("斑图形成动力学", "斑图形成"),
    ("从离散卫星动力学", "从离散卫星行为"),
    ("PDE动力学", "PDE驱动机制"),
    ("动力学", "演化"),  # fallback - catch remaining instances
]

for old, new in replacements_dynamics:
    content = content.replace(old, new)

# 2. Remove parenthetical explanations - convert "(Section~\ref{...})" to "，见Section~\ref{...}"
# But be careful with LaTeX math parentheses
content = content.replace(
    "(ncores 数量确定，k-means", 
    "：ncores 数量确定，k-means"
)
content = content.replace(
    "((1) 26 邻点连通分量标记（C++ PDE）、(2) 基于网格的密度估计（Python CBDP 对比评估）和(3) k-means（ns-3）)",
    "：26 邻点连通分量标记在C++ PDE中实现，基于网格的密度估计在Python CBDP中实现，k-means在ns-3中实现"
)
content = content.replace(
    "（C++ PDE）、", 
    "在C++ PDE中实现，"
)
content = content.replace(
    "（Python CBDP 对比评估）和", 
    "在Python CBDP中实现，"
)
content = content.replace(
    "（ns-3）",
    "在ns-3中实现"
)

# 3. Remove "(相对误差 < 10^{-5})" → "，相对误差低于10^{-5}"
content = content.replace("（相对误差 < 10^{-5}）", "，相对误差低于10^{-5}")
content = content.replace("(相对误差 < 10^{-5})", "，相对误差低于10^{-5}")

# 4. Remove "(Section~\ref{...})" → "，见Section~\ref{...}"
content = re.sub(r'\(Section~\\ref\{([^}]+)\}\)', r'，见Section~\\ref{\1}', content)

# 5. Remove "（表III）" → "，如表III所示"
content = re.sub(r'（表\$\~\\ref\{([^}]+)\}\$）', r'，见\$\~\\ref{\1}\$', content)

# 6. Remove "（图X）" → "，如图X所示"
content = re.sub(r'（图\\ref\{([^}]+)\}）', r'，如图\\ref{\1}所示', content)

# 7. Fix "（$\gamma$ = 0.5, 0.8, 1.0, 1.5, 2.0）" → ": 0.5, 0.8, 1.0, 1.5, 2.0"
content = content.replace(
    "（$\gamma$ = 0.5, 0.8, 1.0, 1.5, 2.0）",
    "：0.5, 0.8, 1.0, 1.5, 2.0"
)
content = content.replace(
    "（0.5, 0.8, 1.0, 1.5, 2.0）",
    "：0.5, 0.8, 1.0, 1.5, 2.0"
)

# 8. Fix "（i）" → "（i）" should stay - these are enumeration
# But "包含：（i）" → "包含三类：（i）"

# 9. Remove "（N = 1,000 和 N = 4,408）" → "，即N = 1,000和N = 4,408"
content = content.replace(
    "（N = 1,000 和 N = 4,408）",
    "，即N = 1,000和N = 4,408"
)

# 10. Remove "（28 核，2.40 GHz，128 GB）" → "（28核，2.40 GHz，128 GB）" → "，配置为28核2.40 GHz处理器和128 GB内存"
content = content.replace(
    "（28 核，2.40 GHz，128 GB）",
    "，配置为28核2.40 GHz处理器和128 GB内存"
)

# 11. Remove "（$dx = 0.5$，$\Delta t = 0.004$，$T = 720$）" → "，其中$dx = 0.5$，$\Delta t = 0.004$，$T = 720$"
content = content.replace(
    "（$dx = 0.5$，$\Delta t = 0.004$，$T = 720$）",
    "，其中$dx = 0.5$，$\Delta t = 0.004$，$T = 720$"
)

# 12. Remove "（3 个主要值 0.1, 0.6, 2.0）" → "：0.1, 0.6, 2.0"
content = content.replace(
    "（3 个主要值 0.1, 0.6, 2.0）",
    "：0.1, 0.6, 2.0"
)

# 13. Remove "（Starlink Gen1 规模）" → "，即Starlink Gen1规模"
content = content.replace(
    "（Starlink Gen1 规模）",
    "，即Starlink Gen1规模"
)

# 14. Remove "（两条面内和两条面间）" → "，包括两条面内和两条面间链路"
content = content.replace(
    "（两条面内和两条面间）",
    "，包括两条面内和两条面间链路"
)

# 15. Remove "（由轨道高度和M决定）" → "，该间距由轨道高度和M决定"
content = content.replace(
    "（由轨道高度和M决定）",
    "，该间距由轨道高度和M决定"
)

# 16. Remove "（纬度高于70°）" → "，即纬度高于70°时"
content = content.replace(
    "（纬度高于70°）",
    "，即纬度高于70°时"
)

# 17. Fix "（Mbps）" → "，单位为Mbps"
content = content.replace("（Mbps）", "，单位为Mbps")

# 18. Remove "（排除60 s预热）" → "，排除60 s预热时间"
content = content.replace("（排除60 s预热）", "，排除60 s预热时间")

# 19. Remove "（MTBF 10,000 s, MTTR 5 s）" → "，MTBF为10,000 s，MTTR为5 s"
content = content.replace(
    "（MTBF 10,000 s, MTTR 5 s）",
    "，MTBF为10,000 s，MTTR为5 s"
)

# 20. Remove "（1,000 数据包缓冲区）" → "，缓冲区容量为1,000个数据包"
content = content.replace(
    "（1,000 数据包缓冲区）",
    "，缓冲区容量为1,000个数据包"
)

# 21. Remove "（$\gamma{=}0.8{\rightarrow}75$, $\gamma{=}1.0{\rightarrow}57$, $\gamma{=}1.5{\rightarrow}197$）" → better
content = content.replace(
    "（$\gamma{=}0.8{\rightarrow}75$，$\gamma{=}1.0{\rightarrow}57$，$\gamma{=}1.5{\rightarrow}197$）",
    "，即$\gamma=0.8$时$n_{\text{cores}}=75$，$\gamma=1.0$时$n_{\text{cores}}=57$，$\gamma=1.5$时$n_{\text{cores}}=197$"
)

# 22. Fix "（Section~\ref{sec:system_model}）" → "，推导见Section~\ref{sec:system_model}"
content = content.replace(
    "（Section~\ref{sec:system_model}）",
    "，推导见Section~\ref{sec:system_model}"
)

# 23. Remove "（Table~\ref{tab:gamma_scan} 和 Fig.~\ref{fig:gamma_scan}）" 
content = re.sub(
    r'（Table~\\ref\{([^}]+)\} 和 Fig\.~\\ref\{([^}]+)\}）',
    r'，见Table~\ref{\1}和Fig.~\ref{\2}',
    content
)

# 24. Remove "（Section~\ref{sec:parameter_analysis}）" → "，见Section~\ref{sec:parameter_analysis}"
content = re.sub(
    r'（Section~\\ref\{([^}]+)\}）',
    r'，见Section~\ref{\1}',
    content
)

# 25. Fix "（非线性正反馈）" → "即非线性正反馈"
content = content.replace(
    "（非线性正反馈）",
    "，即非线性正反馈"
)

# 26. Fix subsection titles
title_replacements = {
    "临界线分析": "临界线：解析预测与C++仿真对比",
    "SNC 数量表征": "SNC数量与$\gamma$：六区域表征",
    "主要结果": "负载不均衡、距离开销与运行时间",
    "ns-3 验证结果": "ns-3验证：吞吐量、丢包率、抖动与延迟",
    "可扩展性分析": "可扩展性：运行时间与内存随星座规模变化",
    "故障恢复": "故障恢复：恢复时间与丢包率",
}

for old_title, new_title in title_replacements.items():
    # Match \subsubsection{old_title}
    content = content.replace(
        f"\\subsubsection{{{old_title}}}",
        f"\\subsubsection{{{new_title}}}"
    )
    # Also match \subsection{old_title}
    content = content.replace(
        f"\\subsection{{{old_title}}}",
        f"\\subsection{{{new_title}}}"
    )

# Also fix the main evaluation subsection title
content = content.replace(
    "\\subsection{对比评估结果}",
    "\\subsection{对比评估：CBDP与现有方法对比}"
)

content = content.replace(
    "\\subsection{消融研究}",
    "\\subsection{消融研究：组件贡献分析}"
)

# Write back
with open(filepath, "w", encoding="utf-8-sig") as f:
    f.write(content)

print("CN v60 sync complete. Changes applied:")
print("- Removed '动力学' → replaced with '演化'/'集中机制'")
print("- Removed 20+ parenthetical explanations")
print("- Fixed subsection titles")
print("- Updated version header")