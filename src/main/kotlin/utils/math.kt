package utils

import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray

fun DoubleArray.toRow(): NDArray<Double, D2> = mk.d2array<Double>(this.size, 1) { this[it] }
fun DoubleArray.toColumn() = mk.d2array<Double>(1, this.size) { this[it] }