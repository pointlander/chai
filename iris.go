// Copyright 2023 The Chai Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"runtime"
	"sort"

	"github.com/pointlander/datum/iris"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// Network is a neural network
type Network struct {
	Theta []float64
	W1    ComplexMatrix
	B1    ComplexMatrix
	W2    ComplexMatrix
	B2    ComplexMatrix
}

// Infer computes the output of the network
func (n Network) Infer(inputs ComplexMatrix) (output ComplexMatrix) {
	l1 := EverettActivation(ComplexAdd(ComplexMul(n.W1, inputs), n.B1))
	l2 := ComplexAdd(ComplexMul(n.W2, l1), n.B2)
	return l2
}

// QRNN implements a complex recurrent neural network for computing a true random string
func IRIS(seed int) {
	cpus := runtime.NumCPU()
	rng := rand.New(rand.NewSource(int64(seed)))
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}
	for i := range datum.Fisher {
		sum := 0.0
		for _, v := range datum.Fisher[i].Measures {
			sum += v * v
		}
		sum = math.Sqrt(sum)
		for j := range datum.Fisher[i].Measures {
			datum.Fisher[i].Measures[j] /= sum
		}
	}
	target := make([][][]float64, 3)
	const points = 3
	target[0] = [][]float64{}
	for i := 0; i < points; i++ {
		target[0] = append(target[0], datum.Fisher[i].Measures)
	}
	target[1] = [][]float64{}
	for i := 0; i < points; i++ {
		target[1] = append(target[1], datum.Fisher[i+50].Measures)
	}
	target[2] = [][]float64{}
	for i := 0; i < points; i++ {
		target[2] = append(target[2], datum.Fisher[i+100].Measures)
	}
	fmt.Println(datum.Fisher[0].Label, datum.Fisher[1].Label)
	fmt.Println(datum.Fisher[50].Label, datum.Fisher[51].Label)
	fmt.Println(datum.Fisher[100].Label, datum.Fisher[101].Label)

	type Distribution struct {
		Mean   float64
		StdDev float64
	}
	const pop = 256
	const cols, rows = 4, 4

	type Genome struct {
		Theta   []Distribution
		W1      []Distribution
		B1      []Distribution
		W2      []Distribution
		B2      []Distribution
		Fitness Stat
		Rank    float64
		Cached  bool
	}
	pool := make([]Genome, 0, pop)

	factor := math.Sqrt(2.0 / float64(cols))
	for i := 0; i < pop; i++ {
		theta := make([]Distribution, 0, 4)
		for i := 0; i < 4; i++ {
			theta = append(theta, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
		}
		w1 := make([]Distribution, 0, 2*cols*rows)
		for i := 0; i < 2*cols*rows; i++ {
			w1 = append(w1, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
		}
		b1 := make([]Distribution, 0, 2*rows)
		for i := 0; i < 2*rows; i++ {
			b1 = append(b1, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
		}
		w2 := make([]Distribution, 0, 2*2*rows*1)
		for i := 0; i < 2*rows*3; i++ {
			w2 = append(w2, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
		}
		b2 := make([]Distribution, 0, 2*1)
		for i := 0; i < 2*1; i++ {
			b2 = append(b2, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
		}
		g := Genome{
			Theta: theta,
			W1:    w1,
			B1:    b1,
			W2:    w2,
			B2:    b2,
		}
		pool = append(pool, g)
	}

	copy := func(g *Genome) Genome {
		theta := make([]Distribution, len(g.Theta))
		copy(theta, g.Theta)
		w1 := make([]Distribution, len(g.W1))
		copy(w1, g.W1)
		b1 := make([]Distribution, len(g.B1))
		copy(b1, g.B1)
		w2 := make([]Distribution, len(g.W2))
		copy(w2, g.W2)
		b2 := make([]Distribution, len(g.B2))
		copy(b2, g.B2)
		return Genome{
			Theta: theta,
			W1:    w1,
			B1:    b1,
			W2:    w2,
			B2:    b2,
		}
	}

	sample := func(rng *rand.Rand, g *Genome) (samples plotter.Values, stats []Stat, network *Network, found bool) {
		stats = make([]Stat, 1)
		scale := 128
		for i := 0; i < scale; i++ {
			n := Network{}
			n.Theta = make([]float64, 0, 4)
			for i := 0; i < 4; i++ {
				n.Theta = append(n.Theta, rng.NormFloat64()*g.Theta[i].StdDev+g.Theta[i].Mean)
			}
			n.W1 = NewComplexMatrix(0, cols, rows)
			for i := 0; i < len(g.W1); i += 2 {
				a := g.W1[i]
				b := g.W1[i+1]
				/*var v complex128
				if rng.NormFloat64()*a.StdDev > a.Mean {
					v = 1
				} else {
					v = -1
				}
				if rng.NormFloat64()*b.StdDev > b.Mean {
					v += 1i
				} else {
					v += -1i
				}
				n.W1.Data = append(n.W1.Data, v)*/
				n.W1.Data = append(n.W1.Data, complex(rng.NormFloat64()*a.StdDev+a.Mean, rng.NormFloat64()*b.StdDev+b.Mean))
			}
			n.B1 = NewComplexMatrix(0, 1, rows)
			for i := 0; i < len(g.B1); i += 2 {
				x := g.B1[i]
				y := g.B1[i+1]
				/*var v complex128
				if rng.NormFloat64()*x.StdDev > x.Mean {
					v = 1
				} else {
					v = -1
				}
				if rng.NormFloat64()*y.StdDev > y.Mean {
					v += 1i
				} else {
					v += -1i
				}
				n.B1.Data = append(n.B1.Data, v)*/
				n.B1.Data = append(n.B1.Data, complex(rng.NormFloat64()*x.StdDev+x.Mean, rng.NormFloat64()*y.StdDev+y.Mean))
			}
			n.W2 = NewComplexMatrix(0, 2*rows, 1)
			for i := 0; i < len(g.W2); i += 2 {
				a := g.W2[i]
				b := g.W2[i+1]
				/*var v complex128
				if rng.NormFloat64()*a.StdDev > a.Mean {
					v = 1
				} else {
					v = -1
				}
				if rng.NormFloat64()*b.StdDev > b.Mean {
					v += 1i
				} else {
					v += -1i
				}
				n.W2.Data = append(n.W2.Data, v)*/
				n.W2.Data = append(n.W2.Data, complex(rng.NormFloat64()*a.StdDev+a.Mean, rng.NormFloat64()*b.StdDev+b.Mean))
			}
			n.B2 = NewComplexMatrix(0, 1, 1)
			for i := 0; i < len(g.B2); i += 2 {
				x := g.B2[i]
				y := g.B2[i+1]
				/*var v complex128
				if rng.NormFloat64()*x.StdDev > x.Mean {
					v = 1
				} else {
					v = -1
				}
				if rng.NormFloat64()*y.StdDev > y.Mean {
					v += 1i
				} else {
					v += -1i
				}
				n.B2.Data = append(n.B2.Data, v)*/
				n.B2.Data = append(n.B2.Data, complex(rng.NormFloat64()*x.StdDev+x.Mean, rng.NormFloat64()*y.StdDev+y.Mean))
			}
			inputs := NewComplexMatrix(0, cols, 1)
			for i := 0; i < cols; i++ {
				inputs.Data = append(inputs.Data, 0)
			}
			fitness := 0.0
			for k, class := range target {
				for _, v := range class {
					for j := range inputs.Data {
						inputs.Data[j] = cmplx.Rect(v[j], n.Theta[j])
					}
					l2 := n.Infer(inputs)
					v := l2.Data[0]
					switch k {
					case 0:
						fit := cmplx.Phase(v) - math.Pi/4
						fitness += fit * fit
					case 1:
						fit := cmplx.Phase(v) - 3*math.Pi/4
						fitness += fit * fit
					case 2:
						fit := cmplx.Phase(v) - 3*math.Pi/2 + 2*math.Pi
						fitness += fit * fit
					}
				}
			}
			fitness /= float64(points * len(target))
			samples = append(samples, fitness)
			stats[0].Add(float64(fitness))
			if fitness <= .01 {
				fmt.Println(i, fitness)
				found = true
				network = &n
				break
			}
		}

		for i := range stats {
			stats[i].Normalize()
		}
		return samples, stats, network, found
	}
	test := func(network *Network) {
		correct, incorrect := 0, 0
		for _, value := range datum.Fisher {
			inputs := NewComplexMatrix(0, cols, 1)
			for i := 0; i < cols; i++ {
				inputs.Data = append(inputs.Data, 0)
			}
			for j := range inputs.Data {
				inputs.Data[j] = cmplx.Rect(value.Measures[j], network.Theta[j])
			}
			l2 := network.Infer(inputs)
			index := 0
			v := l2.Data[0]
			if real(v) > 0 && imag(v) > 0 {
				index = 0
			} else if real(v) < 0 && imag(v) > 0 {
				index = 1
			} else if (real(v) < 0 && imag(v) < 0) || (real(v) > 0 && imag(v) < 0) {
				index = 2
			}
			if index != iris.Labels[value.Label] {
				fmt.Println(value.Label)
				incorrect++
			} else {
				correct++
			}
		}
		fmt.Println("correct", correct, float64(correct)/float64(correct+incorrect))
		fmt.Println("incorrect", incorrect, float64(incorrect)/float64(correct+incorrect))
	}
	done := false
	d := make(plotter.Values, 0, 8)
	for i := range pool {
		dd, stats, network, found := sample(rng, &pool[i])
		fmt.Println(i, stats[0].Mean, stats[0].StdDev)
		if found {
			test(network)
			done = true
			break
		}
		pool[i].Fitness = stats[0]
		pool[i].Cached = true
		d = append(d, dd...)
	}

	p := plot.New()
	p.Title.Text = "iris"

	histogram, err := plotter.NewHist(d, 10)
	if err != nil {
		panic(err)
	}
	p.Add(histogram)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "iris.png")
	if err != nil {
		panic(err)
	}

	rngs, generation := make(map[int]*rand.Rand), 0
Search:
	for !done {
		/*graph := pagerank.NewGraph64()
		for i := range pool {
			for j := i + 1; j < len(pool); j++ {
				// http://homework.uoregon.edu/pub/class/es202/ztest.html
				avga := pool[i].Fitness.Mean
				avgb := pool[j].Fitness.Mean
				avg := avga - avgb
				if avg < 0 {
					avg = -avg
				}
				stddeva := pool[i].Fitness.StdDev
				stddevb := pool[j].Fitness.StdDev
				stddev := math.Sqrt(stddeva*stddeva + stddevb*stddevb)
				z := stddev / avg
				graph.Link(uint64(i), uint64(j), z)
				graph.Link(uint64(j), uint64(i), z)
			}
		}
		graph.Rank(0.85, 0.000001, func(node uint64, rank float64) {
			pool[node].Rank = rank
		})*/
		sort.Slice(pool, func(i, j int) bool {
			return pool[i].Fitness.Mean < pool[j].Fitness.Mean
			//return pool[i].Rank > pool[j].Rank
		})
		pool = pool[:pop]
		fmt.Println(generation, pool[0].Fitness.Mean, pool[0].Fitness.StdDev)
		if pool[0].Fitness.Mean < 1e-32 {
			break Search
		}
		for i := 0; i < pop/4; i++ {
			for j := 0; j < pop/4; j++ {
				if i == j {
					continue
				}
				g := copy(&pool[i])
				w1 := pool[j].W1
				b1 := pool[j].B1
				w2 := pool[j].W2
				b2 := pool[j].B2
				g.W1[rng.Intn(len(g.W1))].Mean = w1[rng.Intn(len(w1))].Mean
				g.W1[rng.Intn(len(g.W1))].StdDev = w1[rng.Intn(len(w1))].StdDev
				g.B1[rng.Intn(len(g.B1))].Mean = b1[rng.Intn(len(b1))].Mean
				g.B1[rng.Intn(len(g.B1))].StdDev = b1[rng.Intn(len(b1))].StdDev
				g.W2[rng.Intn(len(g.W2))].Mean = w2[rng.Intn(len(w2))].Mean
				g.W2[rng.Intn(len(g.W2))].StdDev = w2[rng.Intn(len(w2))].StdDev
				g.B2[rng.Intn(len(g.B2))].Mean = b2[rng.Intn(len(b2))].Mean
				g.B2[rng.Intn(len(g.B2))].StdDev = b2[rng.Intn(len(b2))].StdDev
				pool = append(pool, g)
			}
		}
		for i := 0; i < pop; i++ {
			g := copy(&pool[i])
			g.W1[rng.Intn(len(g.W1))].Mean += rng.NormFloat64()
			g.W1[rng.Intn(len(g.W1))].StdDev += rng.NormFloat64()
			g.B1[rng.Intn(len(g.B1))].Mean += rng.NormFloat64()
			g.B1[rng.Intn(len(g.B1))].StdDev += rng.NormFloat64()
			g.W2[rng.Intn(len(g.W2))].Mean += rng.NormFloat64()
			g.W2[rng.Intn(len(g.W2))].StdDev += rng.NormFloat64()
			g.B2[rng.Intn(len(g.B2))].Mean += rng.NormFloat64()
			g.B2[rng.Intn(len(g.B2))].StdDev += rng.NormFloat64()
			pool = append(pool, g)
		}
		done := make(chan bool, 8)
		i, flight := 0, 0
		task := func(rng *rand.Rand, i int) {
			_, stats, network, found := sample(rng, &pool[i])
			if found {
				test(network)
				done <- true
			}
			pool[i].Fitness = stats[0]
			pool[i].Cached = true
			done <- false
		}
		for i < len(pool) && flight < cpus {
			if pool[i].Cached {
				i++
				continue
			}
			r := rngs[i]
			if r == nil {
				r = rand.New(rand.NewSource(rng.Int63()))
				rngs[i] = r
			}
			go task(r, i)
			i++
			flight++
		}
		for i < len(pool) {
			if pool[i].Cached {
				i++
				continue
			}

			if <-done {
				break Search
			}
			flight--

			r := rngs[i]
			if r == nil {
				r = rand.New(rand.NewSource(rng.Int63()))
				rngs[i] = r
			}
			go task(r, i)
			i++
			flight++
		}
		for flight > 0 {
			<-done
			flight--
		}
		generation++
	}
}
