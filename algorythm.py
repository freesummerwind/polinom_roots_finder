from typing import Dict, List
from multiprocessing import Pool


def get_next_element(element: str, polinom: List[int]) -> str:
    """
    Функция для поиска следующей степени образующего элемента поля
    :param element: str, элемент поля
    :param polinom: List[int], неприводимый полином, корень которого образует поле
    :return: str, следующий элемент поля
    """
    new_element = element[1:] + '0'
    if element[0] == '0':
        return new_element
    for i in range(len(new_element)):
        if polinom[i + 1] == 1:
            new_element = new_element[:i] + str(int(new_element[i] == '0')) + new_element[i+1:]
    return new_element


def build_field_2_m(polinom: List[int]) -> (Dict[int, str], Dict[str, int]):
    """
    Функция, которая строит поле порядка 2^m по неприводимому элементу поля
    :param polinom: List[int], неприводимый полином степени m
    :return: Dict[int, str], отображение степени образующего элемента в элемент поля,
    и Dict[str, int], отображение элемента в соответствующую ему степень образующего элемента
    """
    m = len(polinom) - 1
    degree_to_element = {-1: '0' * (m)}
    element_to_degree = {'0' * (m): -1}
    element = '0' * (m - 1) + '1'
    degree = 0
    while element not in element_to_degree:
        degree_to_element[degree] = element
        element_to_degree[element] = degree
        element = get_next_element(element, polinom)
        degree += 1
    if degree + 1 != 2 ** m:
        raise ValueError('Incorrect polinom')
    return degree_to_element, element_to_degree


class Field:
    """
    Класс, представляющий собой поле порядка 2^m
    """
    def __init__(self, polinom):
        """
        Конструктор класса
        :param polinom: неприводимый полином степени m
        """
        self._polinom = polinom
        self._degree_to_element, self._element_to_degree = build_field_2_m(polinom)
        self._module = len(self._degree_to_element) - 1

    def get_polinom(self) -> List[int]:
        """
        Функция, возвращающая полином, вычетами которого образовано поле
        :return: List[int], полином
        """
        return self._polinom

    def get_polinom_degree(self) -> int:
        """
        Функция, возвращающая степень образующего полинома
        :return: int, степень полинома
        """
        return len(self._polinom) - 1

    def get_element_for_degree(self, degree: int):
        """
        Функция, возвращающая элемент поля, соответствующий некоторой степени образующего элемента
        :param degree: int, степень образующего элемента
        :return: str, элемент поля
        """
        if degree not in self._degree_to_element:
            raise ValueError('No such degree')
        return self._degree_to_element[degree]

    def get_degree_for_element(self, element: str):
        """
        Функция, возвращающая степень образующего элемента, соответствующую некоторому элементу поля
        :param element: str, элемент поля
        :return: int, степень образующего элемента
        """
        if element not in self._element_to_degree:
            raise ValueError('No such element')
        return self._element_to_degree[element]

    def get_inverse_degree(self, degree: int) -> int:
        """
        Функция, возвращающая обратный элемент для данного элемента поля. Элементы представлены степенями образующего элемента
        :param degree: int, обращаемый элемент поля
        :return: int, обратный к данному элементу поля
        """
        return (self._module - degree) % self._module

    def get_inverse_element(self, element: str) -> str:
        """
        Функция, возвращающая обратный элемент для данного элемента поля
        :param degree: str, обращаемый элемент поля, представленный степенью образующего элемента
        :return: str, обратный к данному элементу поля, представленный степенью образующего элемента
        """
        return self._degree_to_element[(self._module - self._element_to_degree[element]) % self._module]

    def sum_of_elements(self, element_1: str, element_2: str) -> str:
        """
        Функция, суммирующая 2 элемента поля
        :param element_1: str, элемент поля
        :param element_2: str, элемент поля
        :return: str, сумма 2 элементов поля
        """
        if element_1 not in self._element_to_degree or element_2 not in self._element_to_degree:
            raise ValueError('Incorrect elements')
        result = ''.join([str(int(element_1[i] != element_2[i])) for i in range(len(element_1))])
        return result

    def sum_of_degrees(self, degree_1: int, degree_2: int):
        """
        Функция, суммирующая 2 элемента поля. Элементы представлены степенями образующего элемента
        :param degree_1: int, элемент поля, представленный степенью образующего элемента
        :param degree_2: int, элемент поля, представленный степенью образующего элемента
        :return: int, сумма 2 элементов поля, представленная степенью образующего элемента
        """
        if degree_1 not in self._degree_to_element or degree_2 not in self._degree_to_element:
            raise ValueError('Incorrect degrees')
        element_1 = self._degree_to_element[degree_1]
        element_2 = self._degree_to_element[degree_2]
        return self._element_to_degree[self.sum_of_elements(element_1, element_2)]

    def mult_of_elements(self, element_1: str, element_2: str):
        """
        Функция, возвращающая произведение 2 элементов поля
        :param element_1: str, элемент поля
        :param element_2: str, элемент поля
        :return: str, произведение 2 элементов поля
        """
        if element_1 not in self._element_to_degree or element_2 not in self._element_to_degree:
            raise ValueError('Incorrect elements')
        degree_1 = self._element_to_degree[element_1]
        degree_2 = self._element_to_degree[element_2]
        if degree_1 == -1 or degree_2 == -1:
            return self._degree_to_element[-1]
        degree = (degree_1 + degree_2) % self._module
        return self._degree_to_element[degree]

    def mult_of_degrees(self, degree_1: int, degree_2: int):
        """
        Функция, возвращающая произведение 2 элементов поля. Элементы представлены степенями образующего элемента
        :param degree_1: int, элемент поля, представленный степенью образующего элемента
        :param degree_2: int, элемент поля, представленный степенью образующего элемента
        :return: int, произведение 2 элементов поля, представленное степенью образующего элемента
        """
        if degree_1 not in self._degree_to_element or degree_2 not in self._degree_to_element:
            raise ValueError('Incorrect degrees')
        if degree_1 == -1 or degree_2 == -1:
            return -1
        return (degree_1 + degree_2) % self._module

def build_field(m: int) -> Field:
    """
    Функция, возвращающая поле порядка 2^m (2 <= m <= 16).
    Внутри функции задана таблица некоторых неприводимых многочленов указанных степеней
    :param m: int, степень порядка поля
    :return: Field, экземпляр класса Field, представляющий собой поле порядка 2^m
    """
    m_to_polinom = {
        2: [1, 1, 1],
        3: [1, 0, 1, 1],
        4: [1, 0, 0, 1, 1],
        5: [1, 0, 0, 1, 0, 1],
        6: [1, 0, 0, 0, 0, 1, 1],
        7: [1, 0, 0, 0, 1, 0, 0, 1],
        8: [1, 0, 0, 0, 1, 1, 1, 0, 1],
        9: [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        10: [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        11: [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        12: [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1],
        13: [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
        14: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
        15: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        16: [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]
    }
    return Field(m_to_polinom[m])


class Polinome:
    """
    Класс, представляющий собой полином над полем F_2^m
    """
    def __init__(self, polinom: List[str], field: Field) -> None:
        """
        Конструктор класса
        :param polinom: List[str], коэффициенты полинома начиная с минимального (при x^0)
        :param field: Field, поле, элементами которого являются коэффициенты полинома
        """
        self._coeffs = polinom
        self._field = field

    def get_degree(self) -> int:
        """
        Функция, возвращающая степень полинома
        :return: int, степень полинома
        """
        return len(self._coeffs) - 1

    def get_coeffs(self) -> List[str]:
        """
        Функция, возвращающая коэффициенты полинома
        :return: List[str], массив элементов полинома
        """
        return self._coeffs

    def get_field(self) -> Field:
        """
        Функция, возвращающая поле, элементами которого являются коэффициенты полинома
        :return: Field, поле
        """
        return self._field

    def __str__(self) -> str:
        """
        Функция, преобразующая полином в строковый вид: a0 + a1*x + a2*x^2 + ..., где а - некий образующий элемент поля
        :return: str, строковый вид полинома
        """
        polinome_parts = []
        for i in range(len(self._coeffs)):
            if self._field.get_degree_for_element(self._coeffs[i]) == -1:
                continue
            elif self._field.get_degree_for_element(self._coeffs[i]) == 0:
                if i == 0:
                    polinome_parts.append('1')
                else:
                    polinome_parts.append(f'x^{i}')
            else:
                if i == 0:
                    polinome_parts.append(f'a^{self._field.get_degree_for_element(self._coeffs[i])}')
                else:
                    polinome_parts.append(f'a^{self._field.get_degree_for_element(self._coeffs[i])} * x^{i}')
        return ' + '.join(polinome_parts)


def mult_element_polinom(element: str, polinom: Polinome, field: Field) -> Polinome:
    """
    Функция, возвращающая полином, умноженный на некоторый элемент поля
    :param element: str, элемент поля
    :param polinom: Polinome, непосредственно многочлен
    :param field: Field, поле, в котором выполняется умножение
    :return: Polinome, полученный многочлен
    """
    if polinom.get_field().get_polinom() != field.get_polinom():
        raise ValueError('different_fields')
    result_polinome = polinom.get_coeffs().copy()
    for i in range(len(result_polinome)):
        result_polinome[i] = field.mult_of_elements(element, result_polinome[i])
    return Polinome(result_polinome, field)


def normalize_polinom(polinom: Polinome) -> Polinome:
    """
    Функция, нормализующая полином (приводящая его к виду, когда старший коэффициент равен 1 поля)
    :param polinom: Polinome, исходный многочлен
    :return: Polinome, полученный многочлен
    """
    max_element = polinom.get_coeffs()[-1]
    inverse = polinom.get_field().get_inverse_element(max_element)
    return mult_element_polinom(inverse, polinom, polinom.get_field())


def sum_polinom_polinom(polinom_1: Polinome, polinom_2: Polinome) -> Polinome:
    """
    Функция, суммирующая 2 многочлена
    :param polinom_1: Polinome, первый исходный многочлен
    :param polinom_2: Polinome, второй исходный многочлен
    :return: Polinome, сумма многочленов
    """
    if polinom_1.get_field().get_polinom() != polinom_2.get_field().get_polinom():
        raise ValueError('different_fields')
    if polinom_1.get_degree() < polinom_2.get_degree():
        polinom_1, polinom_2 = polinom_2, polinom_1
    result_polinome = polinom_1.get_coeffs().copy()
    for i in range(polinom_2.get_degree() + 1):
        result_polinome[i] = polinom_2.get_field().sum_of_elements(polinom_2.get_coeffs()[i], result_polinome[i])
    max_degree = len(result_polinome) - 1
    field = polinom_1.get_field()
    while result_polinome[max_degree] == field.get_element_for_degree(-1) and max_degree > 0:
        max_degree -= 1
    return Polinome(result_polinome[:max_degree + 1], polinom_2.get_field())


def mod_polinom_polinom(polinom_1: Polinome, polinom_2: Polinome) -> Polinome:
    """
    Функция, приводящая первый полином по модулю второго полинома
    :param polinom_1: Polinome, многочлен, который нужно привести по модулю
    :param polinom_2: Polinome, многочлен, по модулю которого будем приводить
    :return: Polinome, полученный многочлен
    """
    if polinom_1.get_field().get_polinom() != polinom_2.get_field().get_polinom():
        raise ValueError('different_fields')
    field = polinom_1.get_field()
    polinom_1 = normalize_polinom(polinom_1)
    polinom_2 = normalize_polinom(polinom_2)
    while polinom_1.get_degree() >= polinom_2.get_degree() and polinom_1.get_coeffs() != [field.get_element_for_degree(-1)]:
        multiplier = field.mult_of_elements(polinom_1.get_coeffs()[-1],
                                            field.get_inverse_element(polinom_2.get_coeffs()[-1]))
        degree_difference = polinom_1.get_degree() - polinom_2.get_degree()
        adding_polinom = Polinome(
            [field.get_element_for_degree(-1) for _ in range(degree_difference)] + polinom_2.get_coeffs(), field)
        adding_polinom = mult_element_polinom(multiplier, adding_polinom, field)
        polinom_1 = sum_polinom_polinom(polinom_1, adding_polinom)
    return polinom_1


def NOD_of_polinomes(polinom_1: Polinome, polinom_2: Polinome) -> Polinome:
    """
    Функция, вычисляющая НОД двух полиномов (реализован алгоритм Евклида)
    :param polinom_1: Polinome, первый исходный многочлен
    :param polinom_2: Polinome, второй исходный многочлен
    :return: Polinome, НОД исходных многочленов
    """
    if polinom_1.get_field().get_polinom() != polinom_2.get_field().get_polinom():
        raise ValueError('different_fields')
    field = polinom_1.get_field()
    if polinom_1.get_degree() < polinom_2.get_degree():
        polinom_1, polinom_2 = polinom_2, polinom_1
    while polinom_2.get_coeffs() != [field.get_element_for_degree(-1)]:
        polinom_1, polinom_2 = polinom_2, mod_polinom_polinom(polinom_1, polinom_2)
    return normalize_polinom(polinom_1)


def build_x_degrees_module_polinom(polinom: Polinome) -> Dict[int, Polinome]:
    """
    Функция, вычисляющая полиномы вида x^(2^k) по модулю переданного полинома. Нужна для построения S(x)
    :param polinom: Polinome, исходный многочлен
    :return: Dict[int, Polinome], отображение степени во взятый по модулю полином
    """
    result = {}
    field = polinom.get_field()
    for i in range(1, polinom.get_degree()):
        result[i] = mod_polinom_polinom(Polinome([field.get_element_for_degree(-1) for _ in range(i)] + [field.get_element_for_degree(0)], field), polinom)
        result[2 * i] = mod_polinom_polinom(Polinome([field.get_element_for_degree(-1) for _ in range(2 * i)] + [field.get_element_for_degree(0)], field), polinom)
    for i in [2 ** k for k in range(field.get_polinom_degree() + 1)]:
        if i not in result:
            old_polinom = result[i // 2]
            current_coeff = old_polinom.get_coeffs()[0]
            current_polinom = Polinome([field.mult_of_elements(current_coeff, current_coeff)], field)
            for j in range(1, old_polinom.get_degree() + 1):
                current_coeff = old_polinom.get_coeffs()[j]
                current_polinom = sum_polinom_polinom(current_polinom,
                                                      mult_element_polinom(field.mult_of_elements(current_coeff, current_coeff),
                                                                           result[2 * j], field))
            result[i] = current_polinom
    return result


def get_s_ax_module_polinom(a_degree: int, x_degrees_module_polinom: Dict[int, Polinome], field: Field) -> Polinome:
    """
    Функция, вычисляющая приведенный по модулю полином S(a^j * x), где a - некий образующий элемент поля
    :param a_degree: int, степень j образующего элемента
    :param x_degrees_module_polinom: Dict[int, Polinome], отображение степени 2^k в приведенные по модулю полиномы вида x^(2^k)
    :param field: Field, поле, в котором ведутся вычисления
    :return: Polinome, полученный многочлен
    """
    a = field.get_element_for_degree(a_degree)
    result = Polinome([field.get_element_for_degree(-1)], field)
    for i in [2 ** k for k in range(field.get_polinom_degree())]:
        result = sum_polinom_polinom(result, mult_element_polinom(a, x_degrees_module_polinom[i], field))
        a = field.mult_of_elements(a, a)
    return result


def find_polinom_roots(polinom: Polinome) -> List[str]:
    """
    Функция, реализующая последовательный алгоритм поиска корней многочлена в поле, над которым он определен
    :param polinom: Polinome, исходный многочлен
    :return: List[str], массив корней многочлена
    """
    if polinom.get_degree() == 1:
        polinom = normalize_polinom(polinom)
        return [polinom.get_coeffs()[0]]
    x_degrees_polinom = build_x_degrees_module_polinom(polinom)
    field = polinom.get_field()
    if x_degrees_polinom[2 ** field.get_polinom_degree()].get_coeffs() != [field.get_element_for_degree(-1), field.get_element_for_degree(0)]:
        polinom = NOD_of_polinomes(polinom, sum_polinom_polinom(x_degrees_polinom[2 ** field.get_polinom_degree()],
                                                                Polinome([field.get_element_for_degree(-1), field.get_element_for_degree(0)], field)))
    if polinom.get_degree() == 0:
        return []
    if polinom.get_degree() == 1:
        polinom = normalize_polinom(polinom)
        return [polinom.get_coeffs()[0]]
    polinomes_in_queue = [polinom]
    result = []
    i = 0
    while polinomes_in_queue:
        s_x = get_s_ax_module_polinom(i, x_degrees_polinom, field)
        s_x_1 = sum_polinom_polinom(s_x, Polinome([field.get_element_for_degree(0)], field))
        new_polinomes_queue = []
        for current in polinomes_in_queue:
            g = NOD_of_polinomes(current, s_x)
            h = NOD_of_polinomes(current, s_x_1)
            if g.get_degree() == 1:
                result.append(g.get_coeffs()[0])
            else:
                new_polinomes_queue.append(g)
            if h.get_degree() == 1:
                result.append(h.get_coeffs()[0])
            else:
                new_polinomes_queue.append(h)
        i += 1
        polinomes_in_queue = new_polinomes_queue
    return result


def find_polinom_roots_n_threads(polinom: Polinome, n: int = 2) -> List[str]:
    """
    Функция, реализующая параллельный алгоритм поиска корней многочлена в поле, над которым он определен
    :param polinom: Polinome, исходный многочлен
    :param n: int, число доступных потоков
    :return: List[str], массив корней многочлена
    """
    if polinom.get_degree() == 1:
        polinom = normalize_polinom(polinom)
        return [polinom.get_coeffs()[0]]
    x_degrees_polinom = build_x_degrees_module_polinom(polinom)
    field = polinom.get_field()
    if x_degrees_polinom[2 ** field.get_polinom_degree()].get_coeffs() != [field.get_element_for_degree(-1), field.get_element_for_degree(0)]:
        polinom = NOD_of_polinomes(polinom, sum_polinom_polinom(x_degrees_polinom[2 ** field.get_polinom_degree()],
                                                                Polinome([field.get_element_for_degree(-1), field.get_element_for_degree(0)], field)))
    if polinom.get_degree() == 0:
        return []
    if polinom.get_degree() == 1:
        polinom = normalize_polinom(polinom)
        return [polinom.get_coeffs()[0]]
    polinomes_in_queue = [polinom]
    result = []
    i = 0
    with Pool(n) as pool:
        while polinomes_in_queue:
            s_x = get_s_ax_module_polinom(i, x_degrees_polinom, field)
            s_x_1 = sum_polinom_polinom(s_x, Polinome([field.get_element_for_degree(0)], field))
            new_polinomes_queue = []
            arguments = []
            for current in polinomes_in_queue:
                arguments.append((current, s_x))
                arguments.append((current, s_x_1))
            nods = pool.starmap(NOD_of_polinomes, arguments)
            for nod in nods:
                if nod.get_degree() == 1:
                    result.append(nod.get_coeffs()[0])
                else:
                    new_polinomes_queue.append(nod)
            i += 1
            polinomes_in_queue = new_polinomes_queue
    return result


if __name__ == '__main__':
    # пример из книги
    field_1 = build_field(6)
    p_1 = Polinome(
        [field_1.get_element_for_degree(13), field_1.get_element_for_degree(49), field_1.get_element_for_degree(43),
         field_1.get_element_for_degree(20), field_1.get_element_for_degree(0)], field_1)
    print(find_polinom_roots(p_1))
    # реальный пример
    field_2 = build_field(8)
    p_2 = Polinome([field_2.get_element_for_degree(0)] + [field_2.get_element_for_degree(-1) for _ in range(254)] + [field_2.get_element_for_degree(0)], field_2)
    print(find_polinom_roots_n_threads(p_2))
    # пример с занятия
    field_3 = build_field(5)
    p_3 = Polinome(['01010', '01101', '11000', '00001', '11011', '01101', '00101', '00011', '11111'], field_3)
    print(find_polinom_roots_n_threads(p_3))


