from manim import *

class clustering(Scene):
    def construct(self):
        t1=Text("各位好，接下来我们将要讲述谱聚类的基本实现")
        self.play(Write(t1))
        self.wait(2)
        self.play(FadeOut(t1))

        t2=Text("大纲")
        self.play(Write(t2))
        self.wait(1)
        self.play(FadeOut(t2))

        p1 = Paragraph(
            "一，前置知识",
            "1.邻接矩阵W",
            "2.度D",
            "3.拉普拉斯矩阵",
            "4.无向图切图\n"
        )
        p2 = Paragraph(
            "二，谱聚类实现",
            "1.优化目标",
            "2.谱聚类切图",
            "3.算法流程"
        )
        self.play(Write(p1))
        self.wait(2)
        self.play(FadeOut(p1))
        self.play(Write(p2))
        self.wait(2)
        self.play(FadeOut(p2))


        t0=Text("一，前置知识")
        self.play(Write(t0))
        self.wait(2)
        self.play(FadeOut(t0))

        #邻接矩阵部分
        t3=Text("邻接矩阵W")
        t3.to_corner(LEFT+UP,buff=SMALL_BUFF)
        self.play(FadeIn(t3))
        self.wait(1)

        #绘制图
        vertices = [1, 2, 3, 4,5]
        edges = [(1, 2), (2, 3), (3, 4), (1, 3), (1, 4), (3, 5)]
        g = Graph(vertices, edges)
        self.play(Create(g))
        self.wait(2)
        self.play(g[1].animate.move_to([1, 1, 0]),
                  g[2].animate.move_to([-1, 1, 0]),
                  g[3].animate.move_to([1, -1, 0]),
                  g[4].animate.move_to([-1, -1, 0]))
        self.wait(2)

        t4=Text("若两点间有边相连接，那么权重大于0，否则为0")
        self.play(g.animate.shift(4*LEFT).scale(1.2))
        t4.shift(2*RIGHT).scale(0.4)
        self.play(FadeIn(t4))
        self.wait(1)
        self.play(FadeOut(t4))

        t5=Text("不妨设权值为0或1，那么左图的邻接矩阵如下")
        t5.shift(2*RIGHT).scale(0.4)
        self.play(FadeIn(t5))
        self.wait(1)
        self.play(FadeOut(t5))

        #绘制邻接矩阵
        m0 = Matrix([[0,1,1,1,0], [1,0, 1,0,0],[1,1,0,1,1],[1,0,1,0,0],[0,0,1,0,0]])
        m0.shift(2*RIGHT)
        self.play(FadeIn(m0))
        self.wait(2)
        self.play(FadeOut(m0), FadeOut(g),FadeOut(t3))
        self.wait(2)

        t6=Text("度D")
        t6.to_corner(LEFT + UP, buff=SMALL_BUFF)
        self.play(FadeIn(t6))
        self.wait(1)

        t7=Text("定义顶点的度为该顶点与其他顶点连接权值之和")
        t8=Tex(r"$d_i=\sum_{i=1}^{N}w_{ij}$")
        t8.shift(2*DOWN)
        self.play(FadeIn(t7))
        self.play(FadeIn(t8))

        self.wait(1)
        self.play(FadeOut(t7))
        self.play(FadeOut(t8))
        self.wait(2)


        t9=Text("由此我们可以定义对角矩阵D")
        t9.shift(2*UP)
        self.play(FadeIn(t9))
        t10=Text("主对角线元素为对应顶点的度数")
        t10.shift(UP)
        self.play(FadeIn(t10))
        t11=Tex(r"$D=diag\{d_1,d_2,...,d_n\}$")
        self.play(FadeIn(t11))
        self.wait(1)
        self.play(FadeOut(t9),FadeOut(t10),FadeOut(t11))

        m1 = Matrix([[3, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 4, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 1]])
        m1.shift(2 * RIGHT)
        self.play(FadeIn(m1), FadeIn(g))
        self.wait(2)
        self.play(FadeOut(g),FadeOut(m1),FadeOut(t6))
        self.wait(1)

        p3=Paragraph(
            "那么自然有一个问题，原始数据大多为多维度数据",
            "并且并没有”边“这类关系",
            "我们应该如何将它们转化为图呢?",
            "常用的转化方式有三种",
            "1.ε邻近法",
            "2.K邻近法",
            "3.全连接法"
        )
        self.play(FadeIn(p3))
        self.wait(2)
        self.play(FadeOut(p3))

        t12=Text("ε邻近法")
        t12.to_corner(LEFT + UP, buff=SMALL_BUFF)
        self.play(FadeIn(t12))
        self.wait(1)

        p3=Paragraph(
            "首先设定一个阈值ε",
            "对于两个数据点i,j",
            "比较它们之间距离(二范数)与ε的关系"
        )
        self.play(FadeIn(p3))
        self.wait(2)
        self.play(FadeOut(p3))
        self.wait(1)

        t13=Tex(r"$s_{ij}=||x_i-x_j||^2_2$")
        t14=Tex(
            r"$s_{ij}<\varepsilon,W_{ij}=\varepsilon$"
        )
        t15 = Tex(
            r"$s_{ij}>\varepsilon,W_{ij}=0$"
        )
        t13.shift(UP)
        t15.shift(DOWN)
        self.play(FadeIn(t13))
        self.play(FadeIn(t14))
        self.play(FadeIn(t15))
        self.wait(2)
        self.play(FadeOut(t13),FadeOut(t14),FadeOut(t15))

        t16=Text("这里我们不妨令ε=2.5")
        t16.to_corner(UP,buff=SMALL_BUFF)
        self.play(FadeIn(t16))

        # 设置点和阈值
        points = [np.array([1, -2, 0]), np.array([3, -2, 0]), np.array([2, 0, 0]), np.array([4, 0, 0])]
        epsilon = 2.2

        # 创建点
        point_mobjects = [Dot(point, color=BLUE) for point in points]

        point_labels = [MathTex(f"P_{{{i + 1}}}").next_to(point, DOWN).scale(0.5) for i, point in enumerate(points)]

        # 添加点和标签到场景
        self.play(*[Create(point) for point in point_mobjects])
        self.play(*[Write(label) for label in point_labels])

        # 计算并展示点之间的距离，并根据阈值绘制边
        lines = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = np.linalg.norm(points[i] - points[j])
                line = Line(points[i], points[j], color=RED if distance <= epsilon else GRAY)
                distance_label = MathTex(f"{distance:.2f}").next_to(line, UP if i % 2 == 0 else DOWN).scale(0.3)
                self.play(Create(line), Write(distance_label))
                lines.append((line, distance_label, distance <= epsilon))

        # 创建邻接矩阵
        adj_matrix = [[0 if np.linalg.norm(points[i] - points[j]) > epsilon else epsilon for j in range(len(points))]
                      for i in range(len(points))]

        # 显示邻接矩阵
        matrix_tex = r"\begin{bmatrix}" + r"\\".join(
            [" & ".join([str(adj_matrix[i][j]) for j in range(len(points))]) for i in
             range(len(points))]) + r"\end{bmatrix}"
        matrix = MathTex(matrix_tex)
        matrix.shift(2*LEFT)

        self.play(Write(matrix))
        self.wait(2)

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)



        #k邻近法
        t17 = Text("k邻近法")
        t17.to_corner(LEFT + UP, buff=SMALL_BUFF)
        self.play(FadeIn(t17))
        self.wait(1)

        # 设置点和参数k
        points = [np.array([-4.5, -2, 0]), np.array([-2.5, -2, 0]), np.array([-3.5, 0, 0]), np.array([-1.5, 1, 0]),
                  np.array([-4.5, 0, 0])]
        k = 2

        # 介绍k邻近法
        intro_text = Text("k近邻法 (k-Nearest Neighbors)").to_edge(UP)
        definition = Text("寻找每个点的k个最近邻居").next_to(intro_text, DOWN)
        self.play(Write(intro_text))
        self.play(Write(definition))
        self.wait(2)
        self.play(FadeOut(intro_text), FadeOut(definition))

        # 创建点
        point_mobjects = [Dot(point, color=BLUE) for point in points]
        point_labels = [MathTex(f"P_{{{i + 1}}}").next_to(point, DOWN).scale(0.4) for i, point in enumerate(points)]

        # 添加点和标签到场景
        for point, label in zip(point_mobjects, point_labels):
            self.play(Create(point), Write(label))

        # 计算并展示点之间的距离
        distances = {}
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = np.linalg.norm(points[i] - points[j])
                distances[(i, j)] = distance
                distances[(j, i)] = distance
                line = Line(points[i], points[j], color=GRAY)
                distance_label = MathTex(f"{distance:.2f}").next_to(line, UP if i % 2 == 0 else DOWN).scale(0.4)
                self.play(Create(line), Write(distance_label))
                self.play(FadeOut(line), FadeOut(distance_label))

        # 确定最近邻居并绘制边
        lines = []
        for i in range(len(points)):
            sorted_neighbors = sorted(range(len(points)), key=lambda j: distances[(i, j)] if i != j else float('inf'))
            for neighbor in sorted_neighbors[:k]:
                line = Line(points[i], points[neighbor], color=RED)
                self.play(Create(line))
                lines.append(line)

        # 创建邻接矩阵
        adj_matrix = np.zeros((len(points), len(points)))
        for i in range(len(points)):
            sorted_neighbors = sorted(range(len(points)), key=lambda j: distances[(i, j)] if i != j else float('inf'))
            for neighbor in sorted_neighbors[:k]:
                adj_matrix[i][neighbor] = distances[(i, neighbor)]

        # 公式和邻接矩阵
        distance_formula = MathTex(r"d(i, j) = \sqrt{\sum_{k=1}^n (x_{ik} - x_{jk})^2}").to_edge(UP)
        matrix_tex = r"\begin{bmatrix}" + r"\\".join(
            [" & ".join([f"{adj_matrix[i][j]:.2f}" if adj_matrix[i][j] != 0 else "0" for j in range(len(points))]) for i
             in range(len(points))]) + r"\end{bmatrix}"
        matrix = MathTex(matrix_tex)
        matrix.next_to(distance_formula, DOWN)
        matrix.shift(2 * RIGHT)

        # 展示公式和邻接矩阵
        self.play(Write(distance_formula), Write(matrix))

        # 等待2秒
        self.wait(2)

        # 移除所有内容
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


        #全连接法
        t18 = Text("全连接法")
        t18.to_corner(LEFT + UP, buff=SMALL_BUFF)
        self.play(FadeIn(t18))
        self.wait(1)

        # 设置点和参数
        points = [np.array([-5.5, -2, 0]), np.array([-3.5, -2, 0]), np.array([-4.5, 0, 0]), np.array([-2.5, 1, 0]),
                  np.array([-5.5, 0, 0])]
        sigma = 1  # 高斯核函数的参数

        # 介绍高斯核函数RBF
        intro_text = Tex("全连接法与高斯核函数 (Fully Connected Method with RBF)",
                         tex_template=TexTemplateLibrary.ctex).to_edge(UP).shift(2 * DOWN)
        definition = Tex("计算每对点之间的RBF核函数值，并生成邻接矩阵", tex_template=TexTemplateLibrary.ctex).next_to(
            intro_text, DOWN)
        self.play(Write(intro_text))
        self.play(Write(definition))
        self.wait(2)
        self.play(FadeOut(intro_text), FadeOut(definition))

        # 创建点
        point_mobjects = [Dot(point, color=BLUE) for point in points]
        point_labels = [MathTex(f"P_{{{i + 1}}}").next_to(point, DOWN).scale(0.3) for i, point in enumerate(points)]

        # 添加点和标签到场景
        for point, label in zip(point_mobjects, point_labels):
            self.play(Create(point), Write(label))

        # 计算并展示点之间的距离和RBF核函数值
        distances = {}
        rbf_values = {}
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = np.linalg.norm(points[i] - points[j])
                distances[(i, j)] = distance
                distances[(j, i)] = distance
                rbf_value = np.exp(-distance ** 2 / (2 * sigma ** 2))
                rbf_values[(i, j)] = rbf_value
                rbf_values[(j, i)] = rbf_value
                line = Line(points[i], points[j], color=GRAY)
                distance_label = MathTex(f"{distance:.2f}").next_to(line, UP if i % 2 == 0 else DOWN).scale(0.3)
                rbf_label = MathTex(f"{rbf_value:.2f}").next_to(line, DOWN if i % 2 == 0 else UP).scale(0.3)
                self.play(Create(line), Write(distance_label), Write(rbf_label))
                self.play(FadeOut(line), FadeOut(distance_label), FadeOut(rbf_label))

        # 创建邻接矩阵
        adj_matrix = np.zeros((len(points), len(points)))
        for i in range(len(points)):
            for j in range(len(points)):
                if i != j:
                    adj_matrix[i][j] = rbf_values[(i, j)]

        # 公式和邻接矩阵
        distance_formula = MathTex(r"d(i, j) = \sqrt{\sum_{k=1}^n (x_{ik} - x_{jk})^2}").to_edge(UP).shift(RIGHT)
        rbf_formula = MathTex(r"\text{RBF}(i, j) = \exp\left(-\frac{d(i, j)^2}{2\sigma^2}\right)").next_to(
            distance_formula, DOWN).shift(RIGHT)
        matrix_tex = r"\begin{bmatrix}" + r"\\".join(
            [" & ".join([f"{adj_matrix[i][j]:.2f}" if adj_matrix[i][j] != 0 else "0" for j in range(len(points))]) for i
             in range(len(points))]) + r"\end{bmatrix}"
        matrix = MathTex(matrix_tex)
        matrix.next_to(rbf_formula, DOWN).shift(0.5 * RIGHT)

        # 展示公式和邻接矩阵
        self.play(Write(distance_formula), Write(rbf_formula), Write(matrix))

        # 等待2秒
        self.wait(2)

        # 移除所有内容
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)

        # 拉普拉斯矩阵
        t19 = Text("拉普拉斯矩阵")
        t19.to_corner(LEFT + UP, buff=SMALL_BUFF)
        self.play(FadeIn(t19))
        self.wait(1)

        p4=Paragraph(
            "拉普拉斯矩阵（Laplacian matrix)）,",
            "也称为基尔霍夫矩阵,",
            " 是表示图的一种矩阵。"
            "给定一个有n个顶点的图,",
            "W,D为其邻接矩阵和度数矩阵",
            "其拉普拉斯矩阵被定义为:"
        )
        self.play(Write(p4))

        t20=Tex(r"L=D-W").align_to(p4, DOWN).shift(DOWN)
        self.play(FadeIn(t20))
        self.wait(2)
        self.play(FadeOut(t20),FadeOut(p4),FadeOut(t19))



        #无向图切图
        t21=Text("无向图切图")
        t21.to_corner(LEFT + UP, buff=SMALL_BUFF)
        self.play(FadeIn(t21))
        self.wait(1)

        # 添加标题
        title = Text("理解无向图的切图", font="思源黑体")
        title.scale(1.2)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # 添加示意图和节点标签
        graph, node_labels = self.draw_graph()
        self.play(Create(graph))
        self.play(Write(node_labels))
        self.wait(1)

        # 解释切图的概念
        explanation1 = Text("在无向图中，切图将图分为两个部分，使得它们不相连。", font="思源黑体", font_size=24)
        explanation1.next_to(graph, DOWN, buff=1)
        self.play(Write(explanation1))
        self.wait(2)

        explanation2 = Text("这些部分称为切割集合（cut-set）。", font="思源黑体", font_size=24)
        explanation2.next_to(explanation1, DOWN, buff=0.5)
        self.play(Write(explanation2))
        self.wait(2)

        # 演示如何进行切图
        cut_line = Line(start=3 * LEFT + 1 * DOWN, end=3 * RIGHT + 1 * UP, color=RED)
        cut_label = Text("切图", font="思源黑体", font_size=24).next_to(cut_line, LEFT)
        self.play(Create(cut_line), Write(cut_label))
        self.wait(2)

        # 显示切图后的效果
        self.play(FadeOut(explanation1), FadeOut(explanation2), FadeOut(cut_line), FadeOut(cut_label),
                  FadeOut(graph), FadeOut(node_labels), FadeOut(title))
        self.wait(1)

        t22 = Text("公式化说明")
        t22.scale(1.2)
        t22.to_edge(UP)
        self.play(FadeIn(t22))
        self.wait(1)

        t23=Text("若A,B为G中的两个子图,那么定义图A,B的切图权重为").next_to(t22,DOWN).shift(DOWN)
        t24=Tex(r"$W(A,B)=\sum_{i \in A,j\in B} w_{ij}$").next_to(t23,DOWN)
        t25=Tex(r"那么对于k个子图的集合$A_1,...,A_k$,我们定义",tex_template=TexTemplateLibrary.ctex).next_to(t24,DOWN)
        t26=Tex(r"$cut(A_1,...,A_k=\frac{1}{2}\sum_{i=1}^k W(A_i,\overline{A_i}))$",tex_template=TexTemplateLibrary.ctex).next_to(t25,DOWN)

        self.play(FadeIn(t23))
        self.play(FadeIn(t24))
        self.play(FadeIn(t25))
        self.play(FadeIn(t26))
        self.wait(2)

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)

        t27=Text("二，谱聚类实现")
        self.play(Write(t27))
        self.wait(1)
        self.play(FadeOut(t27))
        self.wait(1)


        t28=Text("优化目标")
        t28.to_corner(LEFT + UP, buff=SMALL_BUFF)
        self.play(FadeIn(t28))
        self.wait(1)


        p5=Paragraph(
            "一方面,",
            "我们希望去最小化上面提及的cut函数",
            "另一方面,",
            "也要避免一些极端情况",
            "(如每个集合中只有一个顶点)"
                     )
        self.play(Write(p5))
        self.wait(2)
        self.play(FadeOut(p5))

        self.play(FadeOut(t28),FadeOut(p5))
        self.wait(1)

        t29 = Text("谱聚类切图")
        t29.to_corner(LEFT + UP, buff=SMALL_BUFF)
        self.play(FadeIn(t29))
        self.wait(1)

        t30=Text("为了解决上述问题,我们引入ratiocut的概念")
        t31=Tex(r"$ratiocut(A_1,...,A_k)=\frac{1}{2}\sum_{i=1}^k \frac{W(A_i,\overline{A_i}))}{|A_i|}$").next_to(t30,DOWN)
        self.play(FadeIn(t30))
        self.play(FadeIn(t31))
        self.wait(1)
        self.play(FadeOut(t30),FadeOut(t31))
        self.wait(1)

        t32 = Tex(r"对于第j个分割得到的集合$A_j$,我们如下定义$h_j$", tex_template=TexTemplateLibrary.ctex).shift(2*UP)
        t33 = Tex(r"当$v_i\in A_j$时,$h_{j}(i)=\frac{1}{|\sqrt{A_j}|}$,否则为0",
                  tex_template=TexTemplateLibrary.ctex).next_to(t32, DOWN)
        t34 = Tex(r"经过一系列代数推到我们可以得到$h_j^T L h_j=ratiocut(A_i,\overline{A_i})$",
                  tex_template=TexTemplateLibrary.ctex).next_to(t33, DOWN)
        self.play(FadeIn(t32), FadeIn(t33), FadeIn(t34))
        self.wait(4)
        self.play(FadeOut(t32), FadeOut(t33), FadeOut(t34))

        t35 = Text("因此我们只需要让下面的函数最小即可").shift(2*UP).scale(0.8)
        t36 = Tex(r"$ratiocut(A_1,...,A_k)=\sum_{i=1}^{k} h_i^T L h_i=tr(H^T L H)$",
                  tex_template=TexTemplateLibrary.ctex).next_to(t35, DOWN)
        self.play(FadeIn(t35))
        self.play(FadeIn(t36))
        t37 = Tex("这里的H由所有的$h_i$组成,并且已经被标准化(即$H^T H=I$)",
                  tex_template=TexTemplateLibrary.ctex).next_to(t36, DOWN)
        t38 = Paragraph("只需找到最小的几个特征值对应的特征向量",
                        "再利用它们对原始数据降维",
                        "最后用k-means对降维后数据聚类即可").next_to(t37, DOWN).scale(0.8)
        self.play(FadeIn(t37))
        self.play(FadeIn(t38))
        self.wait(3)

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


        #算法流程
        t39 = Text("算法流程")
        t39.to_corner(LEFT + UP, buff=SMALL_BUFF)
        self.play(FadeIn(t39))
        self.wait(1)

        t40=Tex(
            "输入：$D(x_1,...,x_n)$",
        tex_template=TexTemplateLibrary.ctex).shift(2*UP)

        t41=Paragraph(
            "1.构建邻接矩阵",
            "2.计算拉普拉斯矩阵特征值",
            "3.选出最小的k1个特征值与对应向量",
            "构成降维矩阵H",
            "并对原始数据降维",
            "利用k-means进行聚类，得到k2个子集"
        ).next_to(t40,DOWN)
        t42=Tex(r"输出:划分的子集$A_1,...,A_{k_2}$",
                tex_template=TexTemplateLibrary.ctex).next_to(t41,DOWN)

        self.play(FadeIn(t40))
        self.play(FadeIn(t41))
        self.play(FadeIn(t42))

        self.wait(4)

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)

        self.play(FadeIn(Text("感谢观看！！！")))

    def draw_graph(self):
        # 创建一个简单的无向图
        node_positions = {
            "A": LEFT,
            "B": LEFT + UP,
            "C": RIGHT + UP,
            "D": RIGHT,
            "E": 2 * RIGHT
        }

        nodes = {name: Dot(point=pos, color=BLUE) for name, pos in node_positions.items()}
        node_labels = VGroup(*[Text(name, font_size=24).next_to(node, UP) for name, node in nodes.items()])

        edges = VGroup(
            Line(nodes["A"].get_center(), nodes["B"].get_center()),
            Line(nodes["B"].get_center(), nodes["C"].get_center()),
            Line(nodes["C"].get_center(), nodes["D"].get_center()),
            Line(nodes["D"].get_center(), nodes["A"].get_center()),
            Line(nodes["A"].get_center(), nodes["C"].get_center()),
            Line(nodes["C"].get_center(), nodes["E"].get_center()),
            Line(nodes["D"].get_center(), nodes["E"].get_center())
        )

        graph = VGroup(*nodes.values(), edges)
        return graph, node_labels

