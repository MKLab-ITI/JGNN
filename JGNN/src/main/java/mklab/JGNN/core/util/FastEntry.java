package mklab.JGNN.core.util;

import java.util.Map;

public class FastEntry<K, V> implements Map.Entry<K, V> {
    private K key;
    private V value;
    
    public FastEntry() {
 
    }

    public FastEntry(K key, V value) {
        this.key = key;
        this.value = value;
    }

    @Override
    public K getKey() {
        return key;
    }

    @Override
    public V getValue() {
        return value;
    }

    @Override
    public V setValue(V value) {
        V oldValue = this.value;
        this.value = value;
        return oldValue;
    }

    public void setKey(K key) {
        this.key = key;
    }

    @Override
    public String toString() {
        return "FastEntry{" +
                "key=" + key +
                ", value=" + value +
                '}';
    }
}