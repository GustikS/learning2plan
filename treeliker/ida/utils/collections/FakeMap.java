/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package ida.utils.collections;

import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

/**
 * Implementation fo interface Map<R,S> which remains empty no-matter what we put into it.
 * @param <R> type of the key-elements
 * @param <S>  type of the value-elements
 * @author admin
 */
public class FakeMap<R,S> implements Map<R,S> {

    private Set<R> keySet = Collections.<R>emptySet();

    private Collection<S> emptyCollection = Collections.<S>emptyList();

    private Set<Map.Entry<R,S>> entrySet = Collections.<Map.Entry<R,S>>emptySet();

    public int size() {
        return 0;
    }

    public boolean isEmpty() {
        return true;
    }

    public boolean containsKey(Object key) {
        return false;
    }

    public boolean containsValue(Object value) {
        return false;
    }

    public S get(Object key) {
        return null;
    }

    public S put(R key, S value) {
        return null;
    }

    public S remove(Object key) {
        return null;
    }

    public void putAll(Map<? extends R, ? extends S> m) {

    }

    public void clear() {

    }

    public Set<R> keySet() {
        return keySet;
    }

    public Collection<S> values() {
        return emptyCollection;
    }

    public Set<Entry<R, S>> entrySet() {
        return entrySet;
    }

}
