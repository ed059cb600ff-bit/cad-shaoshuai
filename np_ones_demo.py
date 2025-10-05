import numpy as np

# 创建一个5x5的全1数组
# np.ones((5,5), np.uint8) 的参数解析：
# 1. (5,5) 表示创建一个5行5列的二维数组
# 2. np.uint8 表示数组中的元素类型为8位无符号整数（取值范围：0-255）
matrix = np.ones((5,5), np.uint8)

print("=== np.ones((5,5), np.uint8) 详细讲解 ===")
print("\n1. 原始5x5全1数组：")
print(matrix)

print("\n2. 数组的基本属性：")
print(f"- 数组形状 (shape): {matrix.shape}")     # 输出(5,5)
print(f"- 数组维度 (ndim): {matrix.ndim}")       # 输出2
print(f"- 数组类型 (dtype): {matrix.dtype}")     # 输出uint8
print(f"- 数组中元素总数: {matrix.size}")        # 输出25

print("\n3. 不同数据类型的ones数组示例：")
# float类型的ones数组
float_matrix = np.ones((3,3), dtype=float)
print("\nfloat类型 (3x3)：")
print(float_matrix)

# int类型的ones数组
int_matrix = np.ones((3,3), dtype=int)
print("\nint类型 (3x3)：")
print(int_matrix)

print("\n=== 函数说明 ===")
print("np.ones((5,5), np.uint8) 函数解析：")
print("1. np.ones(): 创建一个全1数组的NumPy函数")
print("2. (5,5): 表示数组的形状，这里创建5行5列的二维数组")
print("3. np.uint8: 指定数组的数据类型")
print("   - uint8表示8位无符号整数")
print("   - 取值范围：0 到 255")
print("   - 常用于图像处理，因为图像像素值通常在0-255范围内")
