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
	_ = datum

	type Distribution struct {
		Mean   float64
		StdDev float64
	}
	const pop = 256
	const cols, rows = 8, 8

	type Genome struct {
		Weights []Distribution
		Bias    []Distribution
		Fitness Stat
		Rank    float64
		Cached  bool
	}
	pool := make([]Genome, 0, pop)
	target := make([]bool, 0, 32)
	target = append(target, false, true, true, true, true, true, true, false)
	target = append(target, true, false, true, false, true, false, false, false)
	target = append(target, true, false, false, false, true, true, false, false)
	//target = append(target, false, true, true, true, false, true, false, false)

	factor := math.Sqrt(2.0 / float64(cols))
	for i := 0; i < pop; i++ {
		weights := make([]Distribution, 0, 2*cols*rows)
		for i := 0; i < 2*cols*rows; i++ {
			weights = append(weights, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
		}
		bias := make([]Distribution, 0, 2*rows)
		for i := 0; i < 2*rows; i++ {
			bias = append(bias, Distribution{Mean: factor * rng.NormFloat64(), StdDev: factor * rng.NormFloat64()})
		}
		g := Genome{
			Weights: weights,
			Bias:    bias,
		}
		pool = append(pool, g)
	}

	copy := func(g *Genome) Genome {
		weights := make([]Distribution, len(g.Weights))
		copy(weights, g.Weights)
		bias := make([]Distribution, len(g.Bias))
		copy(bias, g.Bias)
		return Genome{
			Weights: weights,
			Bias:    bias,
		}
	}

	sample := func(rng *rand.Rand, g *Genome) (samples plotter.Values, stats []Stat, found bool) {
		stats = make([]Stat, 5)
		scale := 128
		for i := 0; i < scale; i++ {
			layer := NewComplexMatrix(0, cols, rows)
			for i := 0; i < len(g.Weights); i += 2 {
				a := g.Weights[i]
				b := g.Weights[i+1]
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
				layer.Data = append(layer.Data, v)
			}
			b := NewComplexMatrix(0, 1, rows)
			for i := 0; i < len(g.Bias); i += 2 {
				x := g.Bias[i]
				y := g.Bias[i+1]
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
				b.Data = append(b.Data, v)
			}
			inputs := NewComplexMatrix(0, cols, 1)
			for i := 0; i < cols; i++ {
				inputs.Data = append(inputs.Data, 0)
			}
			correct := 0
			for k, v := range target {
				outputs := ComplexAdd(ComplexMul(layer, inputs), b)
				if (v && real(outputs.Data[0]) > 0 && imag(outputs.Data[0]) > 0) ||
					(v && real(outputs.Data[0]) < 0 && imag(outputs.Data[0]) < 0) ||
					(!v && real(outputs.Data[0]) > 0 && imag(outputs.Data[0]) < 0) ||
					(!v && real(outputs.Data[0]) < 0 && imag(outputs.Data[0]) > 0) {
					correct++
				}
				if k == 1 {
					stats[1].Add(real(outputs.Data[0]))
					stats[2].Add(imag(outputs.Data[0]))
				}
				for j := range outputs.Data {
					var v complex128
					if real(outputs.Data[j]) > 0 {
						v = 1
					} else {
						v = -1
					}
					if imag(outputs.Data[j]) > 0 {
						v += 1i
					} else {
						v += -1i
					}
					outputs.Data[j] = v
				}
				if k == 1 {
					stats[3].Add(real(outputs.Data[0]))
					stats[4].Add(imag(outputs.Data[0]))
				}
				inputs = outputs
			}
			samples = append(samples, float64(correct))
			stats[0].Add(float64(len(target) - correct))
			if correct == len(target) {
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
	p.Title.Text = "rnn"

	histogram, err := plotter.NewHist(d, 10)
	if err != nil {
		panic(err)
	}
	p.Add(histogram)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "qrnn.png")
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
				w := pool[j].Weights
				b := pool[j].Bias
				g.Weights[rng.Intn(len(g.Weights))].Mean = w[rng.Intn(len(w))].Mean
				g.Weights[rng.Intn(len(g.Weights))].StdDev = w[rng.Intn(len(w))].StdDev
				g.Bias[rng.Intn(len(g.Bias))].Mean = b[rng.Intn(len(b))].Mean
				g.Bias[rng.Intn(len(g.Bias))].StdDev = b[rng.Intn(len(b))].StdDev
				pool = append(pool, g)
			}
		}
		for i := 0; i < pop; i++ {
			g := copy(&pool[i])
			g.Weights[rng.Intn(len(g.Weights))].Mean += rng.NormFloat64()
			g.Weights[rng.Intn(len(g.Weights))].StdDev += rng.NormFloat64()
			g.Bias[rng.Intn(len(g.Bias))].Mean += rng.NormFloat64()
			g.Bias[rng.Intn(len(g.Bias))].StdDev += rng.NormFloat64()
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
