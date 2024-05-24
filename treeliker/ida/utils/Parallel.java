/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package ida.utils;

import java.util.*;

import ida.utils.tuples.*;

/**
 * Class for easy parallelization providing several useful methods for dividing work between
 * several threads and then waiting until they finish their jobs.
 * 
 * @author admin
 */
public class Parallel {

    private boolean stop = false;

    private final List<WorkerThread> workers = Collections.synchronizedList(new ArrayList<WorkerThread>());

    private final List<Pair<Runnable,Int>> tasks = Collections.synchronizedList(new ArrayList<Pair<Runnable,Int>>());

    /**
     * Creates a new instance of class Parallel with specified number of threads.
     * @param threadCount the number of threads to be used
     */
    public Parallel(int threadCount){
        for (int i = 0; i < threadCount; i++){
            WorkerThread worker = new WorkerThread();
            workers.add(worker);
            worker.start();
        }
    }

    /**
     * Stops the threads (can take a while because all threads must finish their
     * "atomic" tasks, therefore it is a good idea to have short "atomic takss") 
     */
    public void stop(){
        this.stop = true;
        tasks.notifyAll();
    }

    /**
     * Runs the tasks given as Runnables in parallel and waits until they are finished.
     * 
     * @param tasks the tasks which should be performed
     */
    public void runTasks(List<? extends Runnable> tasks){
        Runnable[] ts = new Runnable[tasks.size()];
        int i = 0;
        for (Runnable task : tasks){
            ts[i] = task;
            i++;
        }
        runTasks(ts);
    }

    /**
     * Runs the tasks given as Runnables in parallel and waits until they are finished.
     * 
     * @param tasks the tasks which should be performed
     */
    public void runTasks(Runnable ...task){
        final Int counter = new Int(task.length);
        for (Runnable r : task){
            synchronized (tasks){
                this.tasks.add(new Pair<Runnable,Int>(r,counter));
                tasks.notify();
            }
        }
        synchronized (counter){
            while (counter.value > 0){
                try {
                    counter.wait();
                } catch (InterruptedException ie){
                    ie.printStackTrace();
                }
            }
        }
    }

    private static class Int {
        
        int value;

        public Int(){}

        public Int(int value){
            this.value = value;
        }
    }

    private class WorkerThread extends Thread {

        public WorkerThread(){
            this.setDaemon(true);
        }

        @Override
        public void run(){
            while (!stop){
                Pair<Runnable,Int> task = null;
                synchronized (tasks){
                    if (tasks.isEmpty()){
                        try {
                            tasks.wait();
                        } catch (InterruptedException ie){
                            ie.printStackTrace();
                        }
                    }
                    if (tasks.size() > 0){
                        task = tasks.remove(tasks.size()-1);
                    }
                }
                if (task != null){
                    task.r.run();
                    synchronized (task.s){
                        task.s.value--;
                        task.s.notify();
                    }
                }
            }
        }
    }

    public void finalize(){
        this.stop = true;
    }
}
