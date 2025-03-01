import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import java.util.*
import kotlin.test.Test

class SimpleNNTest {

    fun createStaticNN() = SimpleNN(intArrayOf(2, 3, 2), false).apply {
        for (index in hiddenLayers.indices) {
            hiddenLayers[index] = hiddenLayers[index].map { 1.0 }
        }
    }

    @Test
    fun predict() {
        val nn = SimpleNN(intArrayOf(2, 1), false)

        nn.printLayers()

        println("-------------------")

        val answers = LinkedList<NDArray<Double, D2>>()
        println(nn.predict(doubleArrayOf(1.0, 1.0), answers).toList())
        println("-------------------")
        answers.forEach { println(); println(it) }


    }

    @Test
    fun predict3() {
        val nn = SimpleNN(intArrayOf(2, 3, 1), true).apply {
            for (index in hiddenLayers.indices) {
                hiddenLayers[index] = hiddenLayers[index].map { 0.0 }
            }
        }

        val task = mk.d2array<Double>(2, 4) { 0.0 }

        nn.printLayers()


        println("-------------------")

        val answers = LinkedList<NDArray<Double, D2>>()
        println(nn.predict(task, answers))
        println("-------------------")
        answers.forEach { println(); println(it) }
//
//        println("-------------------")
//        println(answers.first.view(0))
//        println(answers.first.view(0 ,1))

    }

    @Test
    fun predict2() {
        val nn = SimpleNN(intArrayOf(2, 4, 3, 2), false)

        nn.hiddenLayers.forEach { println(it) }

        println("-------------------")

        val answers = LinkedList<NDArray<Double, D2>>()
        println(nn.predict(doubleArrayOf(1.0, 1.0), answers).toList())
        println("-------------------")
        answers.forEach {
            println(it)
            println()
        }


    }

    @Test
    fun backPropagateErrors() {
        val nn = SimpleNN(intArrayOf(2, 3, 1), true).apply {
            for (index in hiddenLayers.indices) {
                hiddenLayers[index] = hiddenLayers[index].map { 0.2 }
            }
        }
        nn.printLayers()

        val answers = LinkedList<NDArray<Double, D2>>()

        val task = mk.d2array<Double>(2, 6) { 0.0 }


        nn.predict(task, answers)

        println("answers")
        answers.forEach {
            println()
            println(it)
        }

        nn.backPropagateErrors(answers, answers.last.transpose(), 1.0)

        nn.printLayers()

        nn.predict(task, answers)

        println("answers")
        answers.forEach {
            println()
            println(it)
        }

    }

    @Test
    fun multitest() {
        val m1: NDArray<Double, D2> = mk.d2array(10, 1) { it.toDouble() }
        val m2 = m1

        println(m1 * m2)


    }

    @Test
    fun testEducation() {
        //выборка с определённой логикой (я уже забыл условия но сеть всеравно даёт правильный ответ :)))
        //МОЖЕТЕ ЗАКОММЕНТИРОВАТЬ НА СВОЙ ВКУС ЛЮБЫЕ ОДНУ ИЛИ ДВЕ ПАРЫ (ОТВЕТ-ВОПРОС), СЕТЬ СПРАВЛЯЕТСЯ.

        val task: NDArray<Double, D2> = mk.ndarray(
            mk[
                mk[1.0, 0.0, 0.0, 0.0],
                mk[0.0, 1.0, 0.0, 0.0],
                mk[1.0, 1.0, 0.0, 0.0],
                mk[0.0, 0.0, 1.0, 0.0],
                mk[1.0, 0.0, 1.0, 0.0],
                mk[0.0, 1.0, 1.0, 0.0],
                mk[1.0, 1.0, 1.0, 0.0],
                mk[0.0, 0.0, 0.0, 1.0],
                //mk[1.0,0.0,0.0,1.0],
                //mk[0.0,1.0,0.0,1.0],
                mk[1.0, 1.0, 0.0, 1.0],
                mk[0.0, 0.0, 1.0, 1.0],
            ]
        )

        //ответы
        val answ1 = mk.ndarray(
            mk[
                mk[1.0],
                mk[0.0],
                mk[0.0],
                mk[0.0],
                mk[0.0],
                mk[1.0],
                mk[1.0],
                mk[1.0],
                //mk[1.0],  //ожидаемый ответ 1
                //mk[0.0],  //ожидаемый ответ 2
                mk[0.0],
                mk[0.0],
            ]
        )




    }


}