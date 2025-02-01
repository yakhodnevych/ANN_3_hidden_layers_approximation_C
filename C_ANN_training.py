import sys
import openpyxl
import numpy as np
from os.path import join, abspath
# навчання штучної нейронної мережі з 3 прихованими шарами,
# # кількість нейронів однакова для всіх прихованих шарів,
# параметри мережі та відповідні навчальні приклади завантажуються з файлу ексель,
# налаштовані матриці ваг ШНМ зберігаються в окремих файлах

# визначення функції для обчислення стандартної логістичної ф-ії активації
betta = 1  #параметр логістичної функції, за замовчуванням = 1, впливає на швидкість навчання

def logistic(x):
    return 1.0 / (1 + np.exp(-x*betta))

# визначення функції для обчислення похідної від ф-ії активації

def logistic_deriv(x):
    return betta*logistic(x) * (1 - logistic(x))

# запис у текстовий файл результатів навчання - матриці ваг W_1, W_1_ab та W_1_bc
#(якщо у файлі є інформація, вона буде стерта)
def MatrixToFile (W_, n_i, n_j, f_name ):
    # створюється повний шлях для відкриття файлу з врахуванням директорії, в якій запущено програму
    # таке повне ім'я зберігається в data_path
    data_path = join('.', 'Data', f_name)
    data_path = abspath(data_path)

    # W_[i_size,j_size]
    f = open(data_path, 'w')
    for i in range (0,n_i-1):
        for j in range(0,n_j-1):
            f.write(str(W_[i][j])+' ')  # запис у файл лише значень масиву через пробіл, коми і дужки не передаються
        f.write(str(W_[i][n_j-1])+'\n')
    for j in range(0,n_j-1):
        f.write(str(W_[n_i-1][j])+' ')
    f.write(str(W_[n_i-1][n_j-1]))  #останній рядок записуємо поза циклом, щоб уникнути переходу на наступний рядок ('\n' - зміщення в новий рядок)
    f.close()

# параметри навчання мережі
epoch_count = 18  # епохи навчання
alpha = 0.002  # значення коефіцієнта швидкості навчання


# зчитування параметрів мережі та навчальних прикладів з файлу даних Ексель
# відкриваємо вказаний файл тільки для читання значень (формули не зчитуємо)

#створюється повний шлях для відкриття файлу з врахуванням директорії, в якій запущено програму
# таке повне ім'я зберігається в data_path
data_path = join('.','Data',"Training_Data.xlsx")
data_path = abspath(data_path)

Training_Data = openpyxl.open(data_path, read_only=True, data_only=True)

# обираємо перший робочий лист в книзі файлу Ексель
sheet_data = Training_Data.worksheets[0]

max_row_data = sheet_data['A3'].value  # кількість навчальних прикладів зчитуємо безпосередньо з файлу (кількість пар мережі вхід-вихід)

# параметри мережі
input_size = sheet_data['A5'].value  # кількість входів
hidden_size = sheet_data['A7'].value  # к-сть нейронів у прихованому шарі (однакова для всіх прихованих шарів)
output_size = sheet_data['A9'].value  # кількість виходів
hidden_layers_c = 3  # к-сть прихованих шарiв

print('Параметри штучної нейронної мережi')
print('кiлькiсть входiв:', input_size)
print('к-сть нейронiв у прихованому шарi:', hidden_size)
print('к-сть прихованих шарiв:', hidden_layers_c)
print('кiлькiсть виходiв:', output_size)
print('кiлькiсть навчальних прикладiв:', max_row_data)

# визначення та ініціалізація матриць вхідних  і вихідних значень мережі
len_ryadok = input_size

characteristics_riverbed = np.zeros(
    shape=(max_row_data, len_ryadok))  # вхід - гідродинамічні характеристики русла Q або V, B, H, I, n
coef_C = np.zeros(shape=(max_row_data))  # вихід - значення коефіцієнта Шезі

# заповненення дійсних матриць вхідних і вихідних значень мережі навчальними даними із файлу

for i in range(max_row_data):
    coef_C[i] = sheet_data[i + 2][input_size + 1].value
    for j in range(len_ryadok):
        characteristics_riverbed[i][j] = sheet_data[i + 2][j + 1].value


# матриці вагових коефіцієнтів:
# W_1 - між вхідним шаром і першим прихованим шаром нейронів (layer_1а),
# W_1_ab - між 1-им і 2-им прихованими шарами (layer_1a та layer_1b),
# W_1_bc - між 2-им і 3-iм прихованими шарами (layer_1b та layer_1c),
#  W_2 - між 3-iм прихованим шаром і виходом мережі
# задаються випадковими значеннями

np.random.seed(1)
W_1 = 0.02 * np.random.random((input_size, hidden_size)) - 0.01
W_1_ab = 0.02 * np.random.random((hidden_size, hidden_size)) - 0.01
W_1_bc = 0.02 * np.random.random((hidden_size, hidden_size)) - 0.01
W_2 = 0.6 * np.random.random((hidden_size, output_size)) - 0.3

# реалізація епох навчання
# (почергового виконання прямого і зворотного ходу)
# для всіх навчальних прикладів за 1 епоху

correct = 0  # к-сть задовільних результатів навчання
total = 0  # к-сть вивчених навчальних прикладів
error_delta = 0.001  # значення малості похибки мережі

for iteration in range(epoch_count):
    layer_2_e2 = 0  # значення суми квадратів різниць
    for i in range(len(characteristics_riverbed)):
        # прямий хід
        layer_0 = characteristics_riverbed[i:i + 1] # вхідний шар
        layer_1a = logistic(np.dot(layer_0, W_1))  # 1-ий прихований  шар
        layer_1b = logistic(np.dot(layer_1a, W_1_ab)) # 2-ий прихований  шар
        layer_1c = logistic(np.dot(layer_1a, W_1_bc))  # 3-iй прихований  шар
        layer_2 = np.dot(layer_1c, W_2)                 # вихідний  шар

        layer_2_e2 += np.sum((layer_2 - coef_C[i:i + 1]) ** 2)

        # зворотний хід
        layer_2_delta = (coef_C[i:i + 1] - layer_2)
        layer_1c_delta = layer_2_delta.dot(W_2.T) * logistic_deriv(layer_1c)
        layer_1b_delta = layer_1c_delta.dot(W_1_bc.T) * logistic_deriv(layer_1b)
        layer_1a_delta = layer_1b_delta.dot(W_1_ab.T) * logistic_deriv(layer_1a)

        W_2 = W_2 + alpha * layer_1c.T.dot(layer_2_delta)
        W_1_bc = W_1_bc + alpha * layer_1b.T.dot(layer_1c_delta)
        W_1_ab = W_1_ab + alpha * layer_1a.T.dot(layer_1b_delta)
        W_1 = W_1 + alpha * layer_0.T.dot(layer_1a_delta)

               # оцінка точності навчання
        if (np.abs(layer_2_delta) < error_delta):
            correct += 1
        total += 1
        if ((iteration == epoch_count -1) and (i % 10 == 9)):
            print("Iteration:" + str(iteration)+", step:"+str(i))
            print("_C_ = " + str(layer_2))
            print("C = " + str(coef_C[i]))
            print("Quadratic Error:" + str(layer_2_e2))
            print("Training Accuracy:" + str(correct * 100 / float(total)))  # точність навчання
    # print()

# перевірка точності прогнозування
# зчитування і завантаження тестової вибірки даних

# обираємо другий робочий лист з тестовими прикладами в книзі файлу Ексель
sheet_data = Training_Data.worksheets[1]

# кількість тестових прикладів зчитуємо безпосередньо з файлу (кількість пар мережі вхід-вихід)
max_row_test = sheet_data['A3'].value

characteristics_riverbed_test = np.zeros(
    shape=(max_row_test, len_ryadok))  # вхід - тестові гідродинамічні характеристики русла Q або V, B, H, I, n
coef_C_test = np.zeros(shape=(max_row_test))  # вихід - тестові значення коефіцієнта Шезі


# заповненення дійсних матриць вхідних і вихідних значень мережі тестовими прикладами
for i in range(max_row_test):
    coef_C_test[i] = sheet_data[i + 2][input_size + 1].value
    for j in range(len_ryadok):
        characteristics_riverbed_test[i][j] = sheet_data[i + 2][j + 1].value

correct = 0 # к-сть задовільних результатів прогнозування
total = 0 # к-сть перевірених тестових прикладів

# обчислення виходів мережі і оцінка точності
for i in range(max_row_test):
    #прямий хід обчислень
    layer_0 = characteristics_riverbed_test[i:i+1]
    layer_1a = logistic(np.dot(layer_0,W_1))
    layer_1b = logistic(np.dot(layer_1a, W_1_ab))
    layer_1c = logistic(np.dot(layer_1b, W_1_bc))
    layer_2 = np.dot(layer_1c,W_2)

    # оцінка точності прогнозування
    if(np.abs(coef_C_test[i:i+1]-layer_2) < error_delta):
        correct += 1
    total += 1
    print( 'приклад ', i, ', тестове С: ', coef_C_test[i], ', обчислене С: ', layer_2 )
print('всього тестових прикладiв: ', total)
print("Test Accuracy:" + str(correct*100 / float(total))) #точність прогнозування на тестових прикладах

Training_Data.close() # закриття файлу даних Excel

# запис у текстовий файл результатів навчання - матриці ваг W_1, W_1_ab та W_2
#(якщо у файлі є інформація, вона буде стерта)

MatrixToFile (W_1, input_size, hidden_size, "weights_matrix_1.txt" )
MatrixToFile (W_1_ab, hidden_size, hidden_size, "weights_matrix_1_ab.txt" )
MatrixToFile (W_1_bc, hidden_size, hidden_size, "weights_matrix_1_bc.txt" )

#створюється повний шлях для відкриття файлу з врахуванням директорії, в якій запущено програму
# таке повне ім'я зберігається в data_path
data_path = join('.','Data',"weights_matrix_2.txt")
data_path = abspath(data_path)

# W_2[hidden_size,1]
f = open(data_path, 'w')
for i in range (0,hidden_size-1):
    for j in range(0,1):
        f.write(str(W_2[i][j])+'\n')
f.write(str(W_2[hidden_size-1][0])) #останній рядок записуємо поза циклом, щоб уникнути переходу на наступний рядок ('\n' - зміщення в новий рядок)
f.close()

print ('Hалаштованi матрицi вагових коефiєнтiв успiшно збережено.')

print('Натиснiть будь-яку клавiшу для завершення програми.')
input()
