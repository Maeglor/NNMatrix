import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.max
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.junit.Test
import java.util.*
import kotlin.math.absoluteValue
import kotlin.math.roundToInt


val task: NDArray<Double, D2> = mk.ndarray(
    mk[
        mk[1.0, 0.0, 0.0, 0.0],//1.0
        mk[0.0, 1.0, 0.0, 0.0],//0.0
        mk[1.0, 1.0, 0.0, 0.0],//0.0
        mk[0.0, 0.0, 1.0, 0.0],//0.0
        mk[1.0, 0.0, 1.0, 0.0],//0.0
        mk[0.0, 1.0, 1.0, 0.0],//1.0
        mk[1.0, 1.0, 1.0, 0.0],//1.0
        mk[0.0, 0.0, 0.0, 1.0],//1.0
        mk[1.0, 0.0, 0.0, 1.0],//1.0
        mk[0.0, 1.0, 0.0, 1.0],//0.0
        mk[1.0, 1.0, 0.0, 1.0],//0.0
        mk[0.0, 0.0, 1.0, 1.0],//0.0
    ]
)

//ответы
val answ = mk.ndarray(
    mk[
        mk[1.0],
        mk[0.0],
        mk[0.0],
        mk[0.0],
        mk[0.0],
        mk[1.0],
        mk[1.0],
        mk[1.0],
        mk[1.0],
        mk[0.0],
        mk[0.0],
        mk[0.0],
    ]
)


class SimpleNNFixedTest {


    @Test
    fun predict() {
        val nn = SimpleNN(intArrayOf(4, 2, 1))
        val sigmas = LinkedList<NDArray<Double, D2>>()

        do {
            val res = nn.predict(
                task, sigmas
            )
            val err = res - answ
            nn.backPropagateErrors(sigmas, err, 0.1)
        } while (err.max()!!.absoluteValue > 0.01)

        nn.printLayers()
        println("nn.predict(task)")
        println(nn.predict(task).map { it.roundToInt() })


    }

    @Test
    fun backPropagateErrors() {

        println(mk.ndarray(mk[1.0, 0.0, 0.0, 0.0]).reshape(4, 4))


    }
}