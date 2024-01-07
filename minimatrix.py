# F/a/ework(for IEEE course final project
# Fan Cheng, 2022

import random
import copy


class Matrix:
    r"""
        自定义的二维矩阵类

        Args:
            data: 一个二维的嵌套列表，表示矩阵的数据。即 data[i][j] 表示矩阵第 i+1 行第 j+1 列处的元素。
                      当参数 data 不为 None 时，应根据参数 data 确定矩阵的形状。默认值: None
            dim: 一个元组 (n, m) 表示矩阵是 n 行 m 列, 当参数 data 为 None 时，根据该参数确定矩阵的形状；
                     当参数 data 不为 None 时，忽略该参数。如果 data 和 dim 同时为 None, 应抛出异常。[raise TypeError?]默认值: None
            init_value: 当提供的 data 参数为 None 时，使用该 init_value 初始化一个 n 行 m 列的矩阵，
                                    即矩阵各元素均为 init_value. 当参数 data 不为 None 时，忽略该参数。 默认值: 0

    Attributes:
            dim: 一个元组 (n, m) 表示矩阵的形状
            data: 一个二维的嵌套列表，表示矩阵的数据

       Examples:
        >>> mat1 = Matrix(dim=(2, 3), init_value=0)
        >>> print(mat1)
        >>> [[0 0 0]
             [0 0 0]]
        >>> mat2 = Matrix(data=[[0, 1], [1, 2], [2, 3]])
        >>> print(mat2)
        >>> [[0 1]
             [1 2]
             [2 3]]
    """

    def __init__(self, data=None, dim=None, init_value=0):
        if data == None and dim == None:
            raise TypeError("Arguments 'data' and 'dim' cannot be None in the meantime.")
        elif data == None:
            self.dim = dim
            self.data = [[init_value] * (dim[1]) for x in range(dim[0])]
        else:
            self.data = data
            if len(data) == 1:
                row = len(data)
                column = len(data[0])
            else:
                row = len(data)
                column = len(data[1])
                for i in data:
                    if column != len(i):
                        raise ValueError("Argument 'data' is not a valid matrix.")
            self.dim = (row, column)

    def shape(self):
        r"""
        返回矩阵的形状 dim
        """
        return self.dim

    def reshape(self, newdim):
        r"""
        将矩阵从(m,n)维拉伸为newdim=(m1,n1)
        该函数不改变 self

        Args:
                newdim: 一个元组 (m1, n1) 表示拉伸后的矩阵形状。如果 m1 * n1 不等于 self.dim[0] * self.dim[1],
                                应抛出异常

        Returns:
                Matrix: 一个 Matrix 类型的返回结果, 表示 reshape 得到的结果
        """
        if newdim[0] * newdim[1] != self.dim[0] * self.dim[1]:
            raise ValueError("Invalid newdim.")
        new_matrix_data = [[0] * newdim[1] for x in range(newdim[0])]
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                index = j + self.dim[1] * i
                m = index // newdim[1]
                n = index % newdim[1]
                new_matrix_data[m][n] = self.data[i][j]
        new_matrix = Matrix(data=new_matrix_data)
        return new_matrix

    def dot(self, other):
        r"""
        矩阵乘法：矩阵乘以矩阵
        按照公式 A[i, j] = \sum_k B[i, k] * C[k, j] 计算 A = B.dot(C)

        Args:
                other: 参与运算的另一个 Matrix 实例

        Returns:
                Matrix: 计算结果

        Examples:
                >>> A = Matrix(data=[[1, 2], [3, 4]])
                >>> A.dot(A)
                >>> [[ 7 10]
                         [15 22]]
        """
        if self.dim[1] != other.dim[0]:
            raise ValueError("The dimensions of the matrices don't match.")
        result = Matrix(dim=(self.dim[0], other.dim[1]))
        for m in range(result.dim[0]):
            for n in range(result.dim[1]):
                temp = 0
                for k in range(self.dim[1]):
                    temp += self.data[m][k] * other.data[k][n]
                result.data[m][n] = temp
        return result

    def T(self):
        r"""
        矩阵的转置

        Returns:
                Matrix: 矩阵的转置

        Examples:
                >>> A = Matrix(data=[[1, 2], [3, 4]])
                >>> A.T()
                >>> [[1 3]
                         [2 4]]
                >>> B = Matrix(data=[[1, 2, 3], [4, 5, 6]])
                >>> B.T()
                >>> [[1 4]
                         [2 5]
                         [3 6]]
        """
        result = Matrix(dim=(self.dim[1], self.dim[0]))
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                result.data[j][i] = self.data[i][j]
        return result

    def sum(self, axis=None):
        r"""
        根据指定的坐标轴对矩阵元素进行求和

        Args:
                axis: 一个整数，或者 None. 默认值: None
                          axis = 0 表示对矩阵进行按列求和，得到形状为 (1, self.dim[1]) 的矩阵
                          axis = 1 表示对矩阵进行按行求和，得到形状为 (self.dim[0], 1) 的矩阵
                          axis = None 表示对矩阵全部元素进行求和，得到形状为 (1, 1) 的矩阵

        Returns:
                Matrix: 一个 Matrix 类的实例，表示求和结果

        Examples:
                >>> A = Matrix(data=[[1, 2, 3], [4, 5, 6]])
                >>> A.sum()
                >>> [[21]]
                >>> A.sum(axis=0)
                >>> [[5 7 9]]
                >>> A.sum(axis=1)
                >>> [[6]
                         [15]]
        """
        if axis == None:
            result = 0
            for i in range(self.dim[0]):
                result += sum(self.data[i])
            return Matrix(data=[[result]])
        elif axis == 0:
            result = [[]]
            for i in range(self.dim[1]):
                temp = 0
                for j in range(self.dim[0]):
                    temp += self.data[j][i]
                result[0].append(temp)
            return Matrix(data=result)
        elif axis == 1:
            result = []
            for i in range(self.dim[0]):
                temp = 0
                for j in range(self.dim[1]):
                    temp += self.data[i][j]
                result.append([temp])
            return Matrix(data=result)
        else:
            raise ValueError("Axis must be 0, 1, or None.")

    def copy(self):
        r"""
        返回matrix的一个备份

        Returns:
                Matrix: 一个self的备份
        """

        data_copy = copy.deepcopy(self.data)
        dim_copy = self.dim
        return Matrix(data=data_copy, dim=dim_copy)

    def Kronecker_product(self, other):
        r"""
        计算两个矩阵的Kronecker积，具体定义可以搜索，https://baike.baidu.com/item/克罗内克积/6282573

        Args:
                other: 参与运算的另一个 Matrix

        Returns:
                Matrix: Kronecke product 的计算结果
        """
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                if j ==0:
                    m0 = other.num_mul(self[i,j])
                else:
                    m0 = m0.mergematrix(other.num_mul(self[i,j]))
            if i == 0:
                m1 = m0
            else:
                m1 = m1.mergematrix(m0,1)
        return m1

    def __getitem__(self, key):
        r"""
        实现 Matrix 的索引功能，即 Matrix 实例可以通过 [] 获取矩阵中的元素（或子矩阵）

        x[key] 具备以下基本特性：
        1. 单值索引
                x[a, b] 返回 Matrix 实例 x 的第 a 行, 第 b 列处的元素 (从 0 开始编号)
        2. 矩阵切片
                x[a:b, c:d] 返回 Matrix 实例 x 的一个由 第 a, a+1, ..., b-1 行, 第 c, c+1, ..., d-1 列元素构成的子矩阵
                特别地, 需要支持省略切片左(右)端点参数的写法, 如 x 是一个 n 行 m 列矩阵, 那么
                x[:b, c:] 的语义等价于 x[0:b, c:m]
                x[:, :] 的语义等价于 x[0:n, 0:m]

        Args:
                key: 一个元组，表示索引

        Returns:
                索引结果，单个元素或者矩阵切片

        Examples:
            >>> x = Matrix(data=[
                        [0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 0, 1]
                    ])
            >>> x[1, 2]
            >>> 6
            >>> x[0:2, 1:4]
            >>> [[1 2 3]
                 [5 6 7]]
            >>> x[:, :2]
            >>> [[0 1]
                 [4 5]
                 [8 9]]
        """
        new_data = copy.deepcopy(self.data)
        if type(key[0]) == int and type(key[1]) == int:
            return self.data[key[0]][key[1]]
        elif type(key[0]) == slice and type(key[1]) == slice:
            new_data = new_data[key[0]]
            for i in range(len(new_data)):
                new_data[i] = new_data[i][key[1]]
            return Matrix(new_data)
        else:
            raise TypeError("Invalid index for class 'Matrix'")

    def __setitem__(self, key, value):
        r"""
        实现 Matrix 的赋值功能, 通过 x[key] = value 进行赋值的功能

        类似于 __getitem__ , 需要具备以下基本特性:
        1. 单元素赋值
                x[a, b] = k 的含义为，将 Matrix 实例 x 的 第 a 行, 第 b 处的元素赋值为 k (从 0 开始编号)
        2. 对矩阵切片赋值
                x[a:b, c:d] = value 其中 value 是一个 (b-a)行(d-c)列的 Matrix 实例
                含义为, 将由 Matrix 实例 x 的第 a, a+1, ..., b-1 行, 第 c, c+1, ..., d-1 列元素构成的子矩阵 赋值为 value 矩阵
                即 子矩阵的 (i, j) 位置赋值为 value[i, j]
                同样地, 这里也需要支持如 x[:b, c:] = value, x[:, :] = value 等省略写法

        Args:
                key: 一个元组，表示索引
                value: 赋值运算的右值，即要赋的值

        Examples:
                >>> x = Matrix(data=[
                                        [0, 1, 2, 3],
                                        [4, 5, 6, 7],
                                        [8, 9, 0, 1]
                                ])
                >>> x[1, 2] = 0
                >>> x
                >>> [[0 1 2 3]
                     [4 5 0 7]
                     [8 9 0 1]]
                >>> x[1:, 2:] = Matrix(data=[[1, 2], [3, 4]])
                >>> x
                >>> [[0 1 2 3]
                         [4 5 1 2]
                         [8 9 3 4]]
        """

        if type(key[0])== int and type(key[1]) == int:
            self.data[key[0]][key[1]] = value
        else:
            if len(self.data[key[0]]) != value.dim[0] or len(self.data[key[0]][0][key[1]]) != value.dim[1]:
                raise ValueError("The dimensions of matrices don't match.")
            else:
                for i in range(value.dim[0]):
                    self.data[key[0]][i][key[1]] = value.data[i]
        return self.data
      
    def __pow__(self, n):
        r"""
        矩阵的n次幂，n为自然数
        该函数应当不改变 self 的内容

        Args:
                n: int, 自然数

        Returns:
            Matrix: 运算结果
        """
        result = self
        if self.dim[1] != self.dim[0]:
            raise ValueError("The matrix should be a square.")
        for i in range(n):
            result = result.dot(self)
        return result

    def __add__(self, other):
        r"""
        两个矩阵相加
        该函数应当不改变 self 和 other 的内容

        Args:
            other: 一个 Matrix 实例

        Returns:
            Matrix: 运算结果
        """
        if self.dim != other.dim:
            raise ValueError("The dimensions of matrices don't match.")
        else:
            result = [[0] * self.dim[1] for x in range(self.dim[0])]
            for i in range(self.dim[0]):
                for j in range(self.dim[1]):
                    result[i][j] = self.data[i][j] + other.data[i][j]
        return Matrix(result)

    def __sub__(self, other):
        r"""
        两个矩阵相减
        该函数应当不改变 self 和 other 的内容

        Args:
            other: 一个 Matrix 实例

        Returns:
            Matrix: 运算结果
        """
        if self.dim != other.dim:
            raise ValueError("The dimensions of matrices don't match.")
        else:
            result = [[0] * self.dim[1] for x in range(self.dim[0])]
            for i in range(self.dim[0]):
                for j in range(self.dim[1]):
                    result[i][j] = self.data[i][j] - other.data[i][j]
        return Matrix(result)

    def __mul__(self, other):
        r"""
        两个矩阵 对应位置 元素  相乘
        注意 不是矩阵乘法dot
        该函数应当不改变 self 和 other 的内容

        Args:
                other: 一个 Matrix 实例

        Returns:
                Matrix: 运算结果

        Examples:
            >>> Matrix(data=[[1, 2]]) * Matrix(data=[[3, 4]])
            >>> [[3 8]]
        """
        if self.dim != other.dim:
            raise ValueError("The dimensions of matrices don't match.")
        else:
            result = [[0] * self.dim[1] for x in range(self.dim[0])]
            for i in range(self.dim[0]):
                for j in range(self.dim[1]):
                    result[i][j] = self.data[i][j] * other.data[i][j]
        return Matrix(result)

    def __len__(self):
        r"""
        返回矩阵元素的数目

        Returns:
            int: 元素数目，即 行数 * 列数
        """
        return self.dim[1] * self.dim[0]

    def __str__(self):
        r"""
        按照
        [[  0   1   4   9  16  25  36  49]
          [ 64  81 100 121 144 169 196 225]
          [256 289 324 361 400 441 484 529]]
         的格式将矩阵表示为一个 字符串
         ！！！ 注意返回值是字符串
        """
        max_digit = 0
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                if len(str(self.data[i][j])) > max_digit:
                    max_digit = len(str(self.data[i][j]))
        lines = ""
        for i in range(self.dim[0]):
            line0 = ""
            for j in range(self.dim[1]):
                element = str(self.data[i][j])
                element0 = " " * (max_digit - len(element)) + element
                if j != self.dim[1] - 1:
                    line0 += element0 + " "
                else:
                    line0 += element0
            line0 = "[" + line0 + "]"
            if i == 0 and self.dim[0] != 1:
                lines += line0 + "\n"
            elif i == 0 and self.dim[0] == 1:
                lines += line0
            elif i != 0 and i != self.dim[0] - 1:
                lines += " " + line0 + "\n"
            else:
                lines += " " + line0
        str_matrix = "[" + lines + "]" + "\n"
        return str_matrix

    def det(self):
        r"""
        计算方阵的行列式。对于非方阵的情形应抛出异常。
        要求: 该函数应不改变 self 的内容; 该函数的时间复杂度应该不超过 O(n**3).
        提示: Gauss消元

        Returns:
                一个 Python int 或者 float, 表示计算结果
        """
        if self.dim[0] != self.dim[1]:
            raise ValueError("It's not a square matirx.")
        else:
            square_matrix = self.copy()
            square_matrix, flag = square_matrix.to_row_echelon_form_for_det()
            ans = 1
            for i in range(square_matrix.dim[0]):
                ans *= square_matrix[i, i]
            if flag % 2 == 1:
                ans *= -1
            return round(ans, 4)
    
    def inverse(self):
        r"""
        计算非奇异方阵的逆矩阵。对于非方阵或奇异阵的情形应抛出异常。
        要求: 该函数应不改变 self 的内容; 该函数的时间复杂度应该不超过 O(n**3).
        提示: Gauss消元

        Returns:
                Matrix: 一个 Matrix 实例，表示逆矩阵
        """
        A = self.copy()
        if A.dim[0] != A.dim[1]:
            raise ValueError("It's not a square matirx.")
        elif A.det() == 0:
            raise ValueError("It's a singular matirx.")
        else:
            E = I(A.dim[0])
            A_E = A.mergematrix(other=E, axis=0)
            A_E.to_row_standard_simplest_form()
            result = A_E[:, A.dim[0]:]
            return result
            
    def rank(self):
        r"""
        计算矩阵的秩
        要求: 该函数应不改变 self 的内容; 该函数的时间复杂度应该不超过 O(n**3).
        提示: Gauss消元

        Returns:
                一个 Python int 表示计算结果
        """
        result = self.copy()
        result.to_row_echelon_form()
        ans = 0
        for i in range(result.dim[0]):
            if not result[i:i + 1, :].is_zero():
                ans += 1
        return ans

    def guass(self):
        '''
        Guass消元法, 给定一个 Matrix 的实例, 
        返回其简化阶梯型(一个 Matrix 的实例)
        '''
        result1 = self.copy()
        return result1.to_row_standard_simplest_form()
        
    def to_row_echelon_form(self):
        self.normalize_rows()
        for i in range(self.dim[0] - 1):
            index1 = 0
            while self[i, index1] == 0:
                index1 += 1
                if index1 > self.dim[1] - 1:
                    break  
            if index1 > self.dim[1] - 1:
                continue
            else:
                for j in range(i + 1, self.dim[0]):
                    self.add_k_row1_to_row2(i + 1, j + 1, -(self[j, index1] / self[i, index1]))
                self.normalize_rows()
    
    def to_row_echelon_form_for_det(self):
        flag = 0
        self, flag = self.normalize_rows_for_det(flag)
        for i in range(self.dim[0] - 1):
            index1 = 0
            while self[i, index1] == 0:
                index1 += 1
                if index1 > self.dim[1] - 1:
                    break  
            if index1 > self.dim[1] - 1:
                continue
            else:
                for j in range(i + 1, self.dim[0]):
                    self.add_k_row1_to_row2(i + 1, j + 1, -(self[j, index1] / self[i, index1]))
                self, flag = self.normalize_rows_for_det(flag=flag)
        return self, flag
    
    def to_row_standard_simplest_form(self):
        self.normalize_rows()
        for i in range(self.dim[0]):
            if self[i:i + 1, :].is_zero():
                break
            else:
                index1 = 0
                while self[i, index1] == 0:
                    index1 += 1
                self.k_times_row(i + 1, 1 / self[i, index1])
                for j in range(self.dim[0]):
                    if j == i:
                        continue
                    else:
                        self.add_k_row1_to_row2(i + 1, j + 1, -self[j,index1])
                self.normalize_rows()
        
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                self[i, j] = round(self[i, j], 4)

    def normalize_rows(self):
        dt = {}
        for i in range(self.dim[0]):
            column_index = 0
            while self[i, column_index] == 0:
                column_index += 1
                if column_index == self.dim[1]:
                    break
            dt[i] = column_index
        for i in range(self.dim[0]):
            for j in range(i + 1, self.dim[0]):
                if dt[i] > dt[j]:
                    temp = dt[i]
                    dt[i] = dt[j]
                    dt[j] = temp
                    self.change_rows(i + 1, j + 1)

    def normalize_rows_for_det(self, flag):
        dt = {}
        for i in range(self.dim[0]):
            column_index = 0
            while self[i, column_index] == 0:
                column_index += 1
                if column_index == self.dim[1]:
                    break
            dt[i] = column_index
        for i in range(self.dim[0]):
            for j in range(i + 1, self.dim[0]):
                if dt[i] > dt[j]:
                    temp = dt[i]
                    dt[i] = dt[j]
                    dt[j] = temp
                    self, flag = self.change_rows_for_det(i + 1, j + 1, flag)
        return self, flag

    def change_rows(self, row1, row2):
        temp = self[row2 - 1:row2, :]
        self[row2 - 1:row2, :] = self[row1 - 1:row1, :]
        self[row1 - 1:row1, :] = temp

    def change_rows_for_det(self, row1, row2, flag):
        self.change_rows(row1=row1, row2=row2)
        flag += 1
        return self, flag

    def k_times_row(self, row, k):
        self[row - 1:row, :] = self[row - 1:row, :].num_mul(k)
    
    def add_k_row1_to_row2(self, row1, row2, k):
        self[row2 - 1:row2, :] += self[row1 - 1:row1, :].num_mul(k)

    def is_zero(self):
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                if self.data[i][j] != 0:
                    return False
        else:
            return True

    def num_mul(self, n):
        """
        矩阵的数乘 
        其中n为int/float类型的实例,
        返回一个Matrix实例 不改变self
        """
        result = Matrix([[0]*self.dim[1] for x in range(self.dim[0])])
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                result[i,j] = self[i,j] * n
        return result

    def mergematrix(self,other,axis=0):
        r"""
        合并两个矩阵, axis为合并方向,
        axis = 0横向合并,axis = 1纵向合并
        默认值为0.不改变self
        """
        temp = copy.deepcopy(self.data)
        match axis:
            case 0 :
                if self.dim[0] != other.dim[0]:
                    raise ValueError("The dimensions of matrices don't match.")
                else:
                    for i in range(self.dim[0]):
                        temp[i].extend(other.data[i])
            case 1 :
                if self.dim[1] != other.dim[1]:
                    raise ValueError("The dimensions of matrices don't match.")
                else:
                    temp += other.data
            case _:
                raise ValueError
        return Matrix(temp)  
    
def I(n):
    """
    return an n*n unit matrix
    """
    zero = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        zero[i][i] = 1
    return Matrix(data=zero)

def narray(dim, init_value=1):  # dim (,,,,,), init为矩阵元素初始值
    r"""
    返回一个matrix，维数为dim，初始值为init_value

    Args:
            dim: Tuple[int, int] 表示矩阵形状
            init_value: 表示初始值，默认值: 1

    Returns:
            Matrix: 一个 Matrix 类型的实例
    """

    res = Matrix(None, dim, init_value)
    return res

def arange(start, end, step):
    r"""
    返回一个1*n 的 narray 其中的元素类同 range(start, end, step)

    Args:
            start: 起始点(包含)
            end: 终止点(不包含)
            step: 步长

    Returns:
            Matrix: 一个 Matrix 实例
    """
    result = [[x for x in range(start, end, step)]]
    return Matrix(data=result)

def zeros(dim):
    r"""
    返回一个维数为dim 的全0 narray

    Args:
            dim: Tuple[int, int] 表示矩阵形状

    Returns:
            Matrix: 一个 Matrix 类型的实例
    """
    zero_matrix = [[0 for x in range(dim[1])] for i in range(dim[0])]
    return Matrix(data=zero_matrix)

def zeros_like(matrix):
    r"""
    返回一个形状和matrix一样 的全0 narray

    Args:
            matrix: 一个 Matrix 实例

    Returns:
            Matrix: 一个 Matrix 类型的实例

    Examples:
            >>> A = Matrix(data=[[1, 2, 3], [2, 3, 4]])
            >>> zeros_like(A)
            >>> [[0 0 0]
                     [0 0 0]]
    """
    rows = matrix.dim[0]
    columns = matrix.dim[1]
    return zeros((rows, columns))

def ones(dim):
    r"""
    返回一个维数为dim 的全1 narray
    类同 zeros
    """
    one_matrix = [[1 for x in range(dim[1])] for i in range(dim[0])]
    return Matrix(data=one_matrix)

def ones_like(matrix):
    r"""
    返回一个维数和matrix一样 的全1 narray
    类同 zeros_like
    """
    rows = matrix.dim[0]
    columns = matrix.dim[1]
    return ones((rows, columns))

def nrandom(dim):
    r"""
    返回一个维数为dim 的随机 narray
    参数与返回值类型同 zeros
    """
    res = Matrix(None, dim)
    for i in range(dim[0]):
        for j in range(dim[1]):
            res.data[i][j] = random.random()
    return res

def nrandom_like(matrix):
    r"""
    返回一个维数和matrix一样 的随机 narray
    参数与返回值类型同 zeros_like
    """
    return nrandom(matrix.dim)

def concatenate(items, axis=0):
    r"""
    将若干矩阵按照指定的方向拼接起来
    若给定的输入在形状上不对应，应抛出异常
    该函数应当不改变 items 中的元素

    Args:
            items: 一个可迭代的对象，其中的元素为 Matrix 类型。
            axis: 一个取值为 0 或 1 的整数，表示拼接方向，默认值 0.
                      0 表示在第0维即行上进行拼接
                      1 表示在第1维即列上进行拼接

    Returns:
            Matrix: 一个 Matrix 类型的拼接结果

    Examples:
            >>> A, B = Matrix([[0, 1, 2]]), Matrix([[3, 4, 5]])
            >>> concatenate((A, B))
            >>> [[0 1 2]
                     [3 4 5]]
            >>> concatenate((A, B, A), axis=1)
            >>> [[0 1 2 3 4 5 0 1 2]]
    """
    temp = items[0]
    if len(items)>1:
        for i in items[1:]:
            temp = temp.mergematrix(i,axis)
    return(temp)

def vectorize(func):
    r"""
    将给定函数进行向量化

    Args:
            func: 一个Python函数

    Returns:
            一个向量化的函数 F: Matrix -> Matrix, 它的参数是一个 Matrix 实例 x, 返回值也是一个 Matrix 实例；
            它将函数 func 作用在 参数 x 的每一个元素上

    Examples:
            >>> def func(x):
                            return x ** 2
            >>> F = vectorize(func)
            >>> x = Matrix([[1, 2, 3],[2, 3, 1]])
            >>> F(x)
            >>> [[1 4 9]
                     [4 9 1]]
            >>>
            >>> @vectorize
            >>> def my_abs(x):
                            if x < 0:
                                    return -x
                            else:
                                    return x
            >>> y = Matrix([[-1, 1], [2, -2]])
            >>> my_abs(y)
            >>> [[1, 1]
                     [2, 2]]
    """

    def F(x):
        res = Matrix(None, x.dim)
        for i in range(res.dim[0]):
            for j in range(res.dim[1]):
                res.data[i][j] = func(x.data[i][j])
        return res

    return F


if __name__ == "__main__":
    print("test here")

    