// Project 1
// Name - Yashasvi Matta
// Id- 1002091131
// Date- 9/18/23
import java.io.*;
import java.util.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.fs.Path;

import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.mapreduce.lib.output.*;

import org.apache.hadoop.util.*;


class Element implements Writable {
    char temp;
    int index;
    Double val;// value of the matrix element

    public Element() { }

    public Element(char temp, int index, Double val) {
        this.temp = temp;
        this.index = index;
        this.val = val;
    }

    public void write(DataOutput op) throws IOException {
        op.writeShort(temp);
        op.writeDouble(val);
        op.writeInt(index);
        
    }

    public void readFields(DataInput ip) throws IOException {
        temp = ip.readChar();
        val = ip.readDouble();
        index = ip.readInt();
        
    }

    public char gettemp() {
        return temp;
    }

    public int getId() {
        return index;
    }

    public Double getVal() {
        return val;
    }
}

class Coordinate implements WritableComparable<Coordinate> {
    public int a;
    public int b;

    public Coordinate() { }

    public Coordinate(int a, int b) {
        this.a = a;
        this.b = b;
    }

    public void write(DataOutput out) throws IOException {
        // Write the 'a' coordinate to the output 
        out.writeInt(a);
        
        // Write the 'b' coordinate to the output s
        out.writeInt(b);
    }
	
    public void readFields(DataInput ip) throws IOException {
        a = ip.readInt();
        b = ip.readInt();
    }

    @Override

    // Compare method for sorting
    public int compareTo(Coordinate compare) {
        if (a != compare.a) {
                return Integer.compare(a, compare.a);
        } else {
                return Integer.compare(b, compare.b);
        }
        }

    @Override
    public String toString() {
        return a + "," + b;
    }
}

public class Multiply {
    // Mapper for the first matrix.
    public static class MapperOneOne extends Mapper<Object, Text, IntWritable, Element> {
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
             // Parsing the string to its data types.
            String input = value.toString();
            String[] parts = input.split(",");
            int a = 0;
            int b = 0;
            Double v = 0.0;
            if (parts.length > 0) {
                a = Integer.parseInt(parts[0]);
            }
            if (parts.length > 1) {
                b = Integer.parseInt(parts[1]);
            }
            if (parts.length > 2) {
                for (int j = 0; j < 5; j++) {
                    v = Double.parseDouble(parts[2]);
                }
            }
            IntWritable newKey = new IntWritable(b);
            Element newValue = new Element((char) 0, a, v);
            context.write(newKey, newValue);
        }
}
// Mapper for the second matrix.    
public static class MapperOneTwo extends Mapper<Object, Text, IntWritable, Element> {
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String input = value.toString();
            String[] parts = input.split(",");
            parts = input.split(",");
            int x = 0;
            int y = 0;
            Double z = 0.0;
            if (parts.length > 0) {
                x = Integer.parseInt(parts[0]);
            }
            if (parts.length > 1) {
                y = Integer.parseInt(parts[1]);
            }
            if (parts.length > 2) {
                for (int j = 0; j < 10; j++) {
                    z = Double.parseDouble(parts[2]);
                }
            }
            IntWritable newKey = new IntWritable(x);
            Element newValue = new Element((char) 1, y, z);
            context.write(newKey, newValue);
        }
}

    //Reducer for matrix multiplication.
    
public static class ReducerOne extends Reducer<IntWritable, Element, Coordinate, DoubleWritable> {
    @Override
    public void reduce(IntWritable key, Iterable<Element> values, Context contxt) throws IOException, InterruptedException {
        // Creates maps that hold matrix values.
        Map<Integer, Double> matrixM = new HashMap<>();
        Map<Integer, Double> matrixN = new HashMap<>();

        // Separate the values into two maps instead of lists ased on temp

        for (Element value : values) {
            if (value.temp == 0) {
                matrixM.put(value.index, value.val);
            } else {
                matrixN.put(value.index, value.val);
            }
        }

        // Iterate over the entries in the maps
        for (Map.Entry<Integer, Double> entryM : matrixM.entrySet()) {
            for (Map.Entry<Integer, Double> entryN : matrixN.entrySet()) {
                double result = entryM.getValue() * entryN.getValue();
                contxt.write(new Coordinate(entryM.getKey(), entryN.getKey()), new DoubleWritable(result));
            }
        }
    }
}

    // Mapper for second job. This mapper just passes the data to the reducer, nothing else
    public static class MapperTwo extends Mapper<Coordinate, DoubleWritable, Coordinate, DoubleWritable> {
        public void map(Coordinate key, DoubleWritable value, Context context) throws IOException, InterruptedException {
            context.write(key, value);
        }
    }

    // Reducer for the second job. This reducer  acts like a aggregate function.
    public static class ReducerTwo extends Reducer<Coordinate, DoubleWritable, Text, DoubleWritable> {
        public void reduce(Coordinate key, Iterable<DoubleWritable> values, Context contxt) throws IOException, InterruptedException {
            double gross = 0;
            for (DoubleWritable val : values) {
                gross += val.get();
            }
            contxt.write(new Text(key.toString()), new DoubleWritable(gross));
        }
    }

    public static void main(String[] args) throws Exception {
        // Job1 
        Job j1 = Job.getInstance();
        j1.setJobName("Multiply1");
        j1.setJarByClass(Multiply.class);
        j1.setOutputKeyClass(Coordinate.class);
        j1.setOutputValueClass(DoubleWritable.class);
        j1.setMapOutputKeyClass(IntWritable.class);
        j1.setMapOutputValueClass(Element.class);
        j1.setReducerClass(ReducerOne.class);
        j1.setOutputFormatClass(SequenceFileOutputFormat.class);

        MultipleInputs.addInputPath(j1, new Path(args[0]), TextInputFormat.class, MapperOneOne.class);
        MultipleInputs.addInputPath(j1, new Path(args[1]), TextInputFormat.class, MapperOneTwo.class);
        FileOutputFormat.setOutputPath(j1, new Path(args[2]));

       j1.waitForCompletion(true);

        // Job2 
        Job j2 = Job.getInstance();
        j2.setJobName("Multiply2");
        j2.setJarByClass(Multiply.class);
        j2.setOutputKeyClass(Text.class);
        j2.setOutputValueClass(DoubleWritable.class);
        j2.setMapOutputKeyClass(Coordinate.class);
        j2.setMapOutputValueClass(DoubleWritable.class);
        j2.setMapperClass(MapperTwo.class);
        j2.setReducerClass(ReducerTwo.class);
        j2.setOutputFormatClass(TextOutputFormat.class);
        MultipleInputs.addInputPath(j2, new Path(args[2]), SequenceFileInputFormat.class, MapperTwo.class);

        FileOutputFormat.setOutputPath(j2, new Path(args[3]));

        j2.waitForCompletion(true);
    }
}