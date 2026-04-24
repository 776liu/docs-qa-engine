from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# 注册中文字体（使用系统自带字体）
pdfmetrics.registerFont(TTFont('SimSun', 'C:/Windows/Fonts/simsun.ttc'))

c = canvas.Canvas(os.path.join(os.path.dirname(__file__), "test_policy.pdf"), pagesize=A4)
width, height = A4

# 标题
c.setFont("SimSun", 16)
c.drawString(50, height - 50, "公司员工手册")

# 正文
c.setFont("SimSun", 12)
y = height - 100
lines = [
    "第一章：考勤制度",
    "",
    "1. 工作时间：周一至周五 9:00-18:00",
    "2. 迟到处理：每月允许3次10分钟内迟到",
    "3. 加班规定：需提前申请，按1.5倍工资计算",
    "",
    "第二章：福利待遇",
    "",
    "1. 五险一金：入职即缴纳",
    "2. 年终奖：根据绩效考核发放1-3个月工资",
    "3. 团建经费：每人每月200元",
    "",
    "第三章：晋升机制",
    "",
    "1. 每半年进行一次绩效评估",
    "2. 连续两次A级可获得晋升资格",
    "3. 技术序列和管理序列双通道发展",
]

for line in lines:
    c.drawString(50, y, line)
    y -= 25

c.save()
print("PDF已生成: docs/test_policy.pdf")
