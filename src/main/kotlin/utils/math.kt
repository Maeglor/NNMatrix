package utils

import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.data.set

fun DoubleArray.toRow(): NDArray<Double, D2> = mk.d2array<Double>(this.size, 1) { this[it] }
fun DoubleArray.toColumn() = mk.d2array<Double>(1, this.size) { this[it] }

//Устанавливает значение для всего столбца матрицы. По умолчанию последний столбец матрицы
fun NDArray<Double, D2>.setColumn(value: Double, column: Int = shape[0] - 1): NDArray<Double, D2> {
    val index = shape.copyOf()
    index[0] = column

    for (i in 0 until shape.last()) {
        index[1] = i
        this[index] = value
    }
    return this
}

//Устанавливает значение для всей строки матрицы. По умолчанию последняя строка матрицы
fun NDArray<Double, D2>.setRow(value: Double, row: Int = shape[1] - 1): NDArray<Double, D2> {
    val index = shape.copyOf()
    index[1] = row

    for (i in 0 until shape.first()) {
        index[0] = i
        this[index] = value
    }

    return this
}

