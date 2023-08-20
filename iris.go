// Copyright 2023 The Chai Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"

	"github.com/pointlander/datum/iris"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// QRNN implements a complex recurrent neural network for computing a true random string
func IRIS(seed int) {
	cpus := runtime.NumCPU()
	rng := rand.New(rand.NewSource(int64(seed)))
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}
	target := make([][]float64, 3)
	target[0] = datum.Fisher[0].Measures
	target[1] = datum.Fisher[50].Measures
	target[2] = datum.Fisher[100].Measures

	type Distribution struct {
		Mean   float64
		StdDev float64
	}
	const pop = 256
	const cols, rows = 4, 4

	type Genome struct {
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
		w1 := make([]Distribution, 0, 2*cols*rows)
		for i := 0; i < 2*cols*rows; i++ {
			w1 = append(w1, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
		}
		b1 := make([]Distribution, 0, 2*rows)
		for i := 0; i < 2*rows; i++ {
			b1 = append(b1, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
		}
		w2 := make([]Distribution, 0, 2*cols*3)
		for i := 0; i < 2*cols*rows; i++ {
			w2 = append(w2, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
		}
		b2 := make([]Distribution, 0, 2*3)
		for i := 0; i < 2*rows; i++ {
			b2 = append(b2, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
		}
		g := Genome{
			W1: w1,
			B1: b1,
			W2: w2,
			B2: b2,
		}
		pool = append(pool, g)
	}

	copy := func(g *Genome) Genome {
		w1 := make([]Distribution, len(g.W1))
		copy(w1, g.W1)
		b1 := make([]Distribution, len(g.B1))
		copy(b1, g.B1)
		w2 := make([]Distribution, len(g.W1))
		copy(w2, g.W2)
		b2 := make([]Distribution, len(g.B2))
		copy(b1, g.B1)
		return Genome{
			W1: w1,
			B1: b1,
			W2: w2,
			B2: b2,
		}
	}

	sample := func(rng *rand.Rand, g *Genome) (samples plotter.Values, stats []Stat, found bool) {
		stats = make([]Stat, 1)
		scale := 128
		for i := 0; i < scale; i++ {
			w1 := NewComplexMatrix(0, cols, rows)
			for i := 0; i < len(g.W1); i += 2 {
				a := g.W1[i]
				b := g.W1[i+1]
				//layer.Data = append(layer.Data, complex((rng.NormFloat64()+a.Mean)*a.StdDev, (rng.NormFloat64()+b.Mean)*b.StdDev))
				var v complex128
				if rng.NormFloat64() > a.Mean {
					v = 1
				} else {
					v = -1
				}
				if rng.NormFloat64() > b.Mean {
					v += 1i
				} else {
					v += -1i
				}
				w1.Data = append(w1.Data, v)
			}
			b1 := NewComplexMatrix(0, 1, rows)
			for i := 0; i < len(g.B1); i += 2 {
				x := g.B1[i]
				y := g.B1[i+1]
				//b.Data = append(b.Data, complex((rng.NormFloat64()+x.Mean)*x.StdDev, (rng.NormFloat64()+y.Mean)*y.StdDev))
				var v complex128
				if rng.NormFloat64() > x.Mean {
					v = 1
				} else {
					v = -1
				}
				if rng.NormFloat64() > y.Mean {
					v += 1i
				} else {
					v += -1i
				}
				b1.Data = append(b1.Data, v)
			}
			w2 := NewComplexMatrix(0, cols, 3)
			for i := 0; i < len(g.W1); i += 2 {
				a := g.W2[i]
				b := g.W2[i+1]
				//layer.Data = append(layer.Data, complex((rng.NormFloat64()+a.Mean)*a.StdDev, (rng.NormFloat64()+b.Mean)*b.StdDev))
				var v complex128
				if rng.NormFloat64() > a.Mean {
					v = 1
				} else {
					v = -1
				}
				if rng.NormFloat64() > b.Mean {
					v += 1i
				} else {
					v += -1i
				}
				w2.Data = append(w2.Data, v)
			}
			b2 := NewComplexMatrix(0, 1, 3)
			for i := 0; i < len(g.B1); i += 2 {
				x := g.B1[i]
				y := g.B1[i+1]
				//b.Data = append(b.Data, complex((rng.NormFloat64()+x.Mean)*x.StdDev, (rng.NormFloat64()+y.Mean)*y.StdDev))
				var v complex128
				if rng.NormFloat64() > x.Mean {
					v = 1
				} else {
					v = -1
				}
				if rng.NormFloat64() > y.Mean {
					v += 1i
				} else {
					v += -1i
				}
				b2.Data = append(b2.Data, v)
			}
			inputs := NewComplexMatrix(0, cols, 1)
			for i := 0; i < cols; i++ {
				inputs.Data = append(inputs.Data, 0)
			}
			correct, incorrect := 0, 0
			for k, v := range target {
				for j := range inputs.Data {
					inputs.Data[j] = complex(v[j], 0)
				}
				l1 := ComplexAdd(ComplexMul(w1, inputs), b1)
				for j := range l1.Data {
					var v complex128
					if real(l1.Data[j]) > 0 {
						v = 1
					} else {
						v = -1
					}
					if imag(l1.Data[j]) > 0 {
						v += 1i
					} else {
						v += -1i
					}
					l1.Data[j] = v
				}
				l2 := ComplexAdd(ComplexMul(w2, l1), b2)
				for j, v := range l2.Data {
					if ((j == k) && real(v) > 0 && imag(v) > 0) ||
						((j == k) && real(v) < 0 && imag(v) < 0) {
						correct++
					} else if ((j != k) && real(v) > 0 && imag(v) < 0) ||
						((j != k) && real(v) < 0 && imag(v) > 0) {
						incorrect++
					}
				}
			}
			samples = append(samples, float64(correct))
			stats[0].Add(float64(len(target) - correct + incorrect))
			if correct == len(target) && incorrect == 0 {
				fmt.Println(i, correct)
				found = true
				break
			}
		}

		for i := range stats {
			stats[i].Normalize()
		}
		return samples, stats, found
	}
	done := false
	d := make(plotter.Values, 0, 8)
	for i := range pool {
		dd, stats, found := sample(rng, &pool[i])
		fmt.Println(i, stats[0].Mean, stats[0].StdDev)
		fmt.Println(stats)
		if found {
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
			_, stats, found := sample(rng, &pool[i])
			if found {
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
