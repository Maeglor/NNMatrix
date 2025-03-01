import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.rand
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.append
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.minusAssign
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import utils.DoNothingMutableCollection
import utils.toColumn
import utils.toRow
import java.util.*
import kotlin.math.exp


class SimpleNN(layers: IntArray, val bias: Boolean = true) {
    val biasConst =
        mk.ndarray(mk[mk[1.0]]) // единичная матрица для добавления сигнала смещения с фиксированным значением

    // Функция активации (сигма)
    fun activationFunction(x: Double): Double {
        return 1.0 / (1.0 + exp(-x))
    }


    init {
        if (layers.size < 2)
            throw Exception("layers size must be >= 2 ")
    }


    //Матрица слоя. Каждый столбец это нейрон.
    val hiddenLayers: Array<NDArray<Double, D2>> = Array(layers.size - 1) { layer ->
        val biasSize = if (bias) 1 else 0
        val biasNeuronSize = if (layer < layers.size - 2) biasSize else 0
        mk.rand<Double>(
            layers[layer + 1] + biasNeuronSize, // Маленькая хитрость: вместо нейрона смещения добавляем еще один полноценный нейрон, ответ которого всегда меняем на 1.
            // Это несколько увеличивает количество операций, но позволяет не выделять память при добавлении слоя к результату.
            // Кроме последнего слоя на котором это не нужно
            layers[layer] + biasSize // если нейрон смещения используется добавляем дендриты

        )
    }

    /**
     *  Печать всех слоев нейросети для анализа.
     */
    fun printLayers() {
        hiddenLayers.forEachIndexed { index, layer ->
            println("layer $index")
            println(layer)
        }
    }

    fun predict(input: DoubleArray): DoubleArray = predict(input, DoNothingMutableCollection())

    fun predict(input: DoubleArray, answers: MutableCollection<NDArray<Double, D2>>): DoubleArray =
        predict(input.toRow(), answers).data.getDoubleArray()


    fun predict(input: NDArray<Double, D2>, answers: MutableCollection<NDArray<Double, D2>>): NDArray<Double, D2> {

        answers.clear()
        var processed = input
        if (bias) {
            processed = processed.append((DoubleArray(processed.shape.last()) { 1.0 }).toColumn(), 0)
        }

        hiddenLayers.forEach { layer ->

            answers.add(processed)
            processed = layer.dot(processed).map { activationFunction(it) }



            if (layer !== hiddenLayers.last())
                processed.setColumn(1.0)
        }

        answers.add(processed)

        return processed
    }

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

    fun backPropagateErrors(sigmaSignals: List<NDArray<Double, D2>>, errors: NDArray<Double, D2>, learnConf: Double) {

        val errorsList = LinkedList<NDArray<Double, D2>>()

        //Вычисляем ошибки в каждом слое, путем обратного их распространения. Просто прогоняем ошибки в обратную сторону без учета сигма-функции
        errorsList.add(errors)
        hiddenLayers.reversed().forEach { layer ->
            errorsList.add(errorsList.last.dot(layer).setRow(0.0))
        }
        // Инвертируем список чтобы порядок ошибок совпадал с номером слоя (мы считали их из конца в начало)
        errorsList.reverse()

        println("errorsList")
        errorsList.forEach {
            println()
            println(it)
        }

        for (index in hiddenLayers.indices) {
            println("layer:$index")
            // Для коррекции весов необходимо воспользоваться формулой:
            // in * learnConf * sigmSignal * (1 - sigmSignal) * error
            // in -- входной сигнал в нейрон
            // learnConf -- коэффициент обучения
            // sigmSignal -- выходной сигнал нейрона (вместе с сигма-функцией)
            // sigmSignal * (1 - sigmSignal) -- производная сигма функции выходного сигнала
            // значение ошибки

            // строку выходных сигналов транспонируем в столбец
            val sigmaErrors = sigmaSignals[index + 1].transpose()
                // преобразуем ее в столбец производных
                .map { it * 1 - it + learnConf } *
                    // и умножаем поэлементно на столбец ошибок
                    errorsList[index + 1]
            // получили столбец из learnConf * sigmSignal * (1 - sigmSignal) * error


            println("sigmaSignals[index]")
            println(sigmaSignals[index])

            //умножаем строку входных индексов на столбец с ошибками, получаем прямоугольную матрицу коррекций, для слоя
            val correctionsMatrix = sigmaSignals[index].dot(sigmaErrors).transpose()

            println("correctionsMatrix")
            println(correctionsMatrix)

            // и наконец вычитаем поэлементно матрицу коррекций из матрицы слоя.
            hiddenLayers[index] -= correctionsMatrix
        }
    }


}