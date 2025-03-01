package utils

class DoNothingMutableCollection<T>: MutableCollection<T> {
    override val size = 0
    override fun clear() {}
    override fun isEmpty(): Boolean  = true
    override fun iterator(): MutableIterator<T> = object :MutableIterator<T>{
        override fun hasNext() = false

        override fun next(): T {
            TODO("Not yet implemented")
        }

        override fun remove() {}
    }
    override fun retainAll(elements: Collection<T>): Boolean  = false
    override fun removeAll(elements: Collection<T>): Boolean  = false
    override fun remove(element: T): Boolean  = false
    override fun containsAll(elements: Collection<T>): Boolean  = false
    override fun contains(element: T): Boolean = false
    override fun addAll(elements: Collection<T>): Boolean = false
    override fun add(element: T): Boolean  = false
}