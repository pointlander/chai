// Copyright 2023 The Chai Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/pointlander/pagerank"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

var (
	// FlagGradient bits mode
	FlagGradient = flag.Bool("gradient", false, "gradient mode")
	// FlagNumbers numbers mode
	FlagNumbers = flag.Bool("numbers", false, "numbers mode")
	// FlagSwarm swarm mode
	FlagSwarm = flag.Bool("swarm", false, "swarm mode")
	// FlagTarget is the target value
	FlagTarget = flag.Int("target", 77, "target value")
)

func main() {
	flag.Parse()

	if *FlagGradient {
		for seed := 1; seed < 1000; seed++ {
			if Gradient(seed) {
				return
			}
		}
		return
	}

	if *FlagNumbers {
		Numbers()
		return
	}

	if *FlagSwarm {
		for seed := 1; seed < 1000; seed++ {
			if Swarm(seed) {
				return
			}
		}
		return
	}

	rnd := rand.New(rand.NewSource(1))
	samples := make(plotter.Values, 0, 8)
	for i := 0; i < 1000; i++ {
		x := 0
		for i := 0; i < 10; i++ {
			x += *FlagTarget % (rnd.Intn(77) + 1)
		}
		samples = append(samples, float64(x))
	}

	p := plot.New()
	p.Title.Text = "residulas"

	histogram, err := plotter.NewHist(samples, 10)
	if err != nil {
		panic(err)
	}
	p.Add(histogram)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "residuals.png")
	if err != nil {
		panic(err)
	}
}

func Gradient(seed int) bool {
	rnd := rand.New(rand.NewSource(int64(seed)))
	type Distribution struct {
		Mean   float64
		StdDev float64
	}
	target := *FlagTarget
	n := int(math.Ceil(math.Log2(float64(target))))
	a := make([]Distribution, 0, n)
	for i := 0; i < n; i++ {
		a = append(a, Distribution{Mean: rnd.NormFloat64(), StdDev: 1})
	}
	fmt.Println(n)
	fmt.Println(a)
	shownum := func(a []Distribution) int {
		x := 0
		e := 1
		for _, v := range a {
			if v.Mean > 0 {
				x += e
			}
			e *= 2
		}
		fmt.Println(x)
		return x
	}
	samples := 8 * 1024
	sample := func(a, b []Distribution) (avg, sd float64) {
		i := 0
		for i < samples {
			x := 0
			y := 0
			e := 1
			k := 0
			for _, v := range a {
				if (rnd.NormFloat64()+v.Mean)*v.StdDev > 0 {
					x += e
				}
				e *= 2
				k++
			}
			e = 1
			k = 0
			for _, v := range b {
				if (rnd.NormFloat64()+v.Mean)*v.StdDev > 0 {
					y += e
				}
				e *= 2
				k++
			}
			xx := 0
			if x > 0 {
				xx = target % x
			}
			yy := 0
			if y > 0 {
				yy = target % y
			}
			cost := yy * xx
			avg += float64(cost)
			sd += float64(cost) * float64(cost)
			i += 1
		}
		avg /= float64(samples)
		sd = math.Sqrt(sd/float64(samples) - avg*avg)
		return avg, sd
	}

	j := 0
	for j < 33 {
		type Sample struct {
			A      []Distribution
			Weight float64
		}
		samples := make([]Sample, 16)
		for i := range samples {
			samples[i].A = make([]Distribution, len(a))
			copy(samples[i].A, a)
			for j := range samples[i].A {
				samples[i].A[j].Mean += rnd.NormFloat64()
				samples[i].A[j].StdDev += rnd.NormFloat64()
				if samples[i].A[j].StdDev < 0 {
					samples[i].A[j].StdDev = -samples[i].A[j].StdDev
				}
			}

		}
		graph := pagerank.NewGraph64()
		for i := range samples {
			remainder := samples[i:]
			for j := range remainder {
				avg, _ := sample(samples[i].A, remainder[j].A)
				graph.Link(uint64(i), uint64(j), float64(target)*float64(target)-avg)
				graph.Link(uint64(j), uint64(i), float64(target)*float64(target)-avg)
			}
		}
		graph.Rank(0.85, 0.000001, func(node uint64, rank float64) {
			samples[node].Weight = rank
		})
		sort.Slice(samples, func(i, j int) bool {
			return samples[i].Weight > samples[j].Weight
		})

		copy(a, samples[0].A)
		x := shownum(a)
		if x == 0 {
			return false
		}
		if target%x == 0 {
			if x == 1 || x == target {
				return false
			} else {
				fmt.Println(target / x)
				return true
			}
		}
		j++
	}
	fmt.Println(a)
	shownum(a)
	return false
}

const (
	// Width is the width of the swarm
	Width = 2
)

func Swarm(seed int) bool {
	fmt.Println(seed)
	rnd := rand.New(rand.NewSource(int64(seed)))
	type Distribution struct {
		Mean   float64
		StdDev float64
	}
	target := *FlagTarget
	max := math.Sqrt(float64(target))
	n := int(math.Ceil(math.Log2(max) + 1))

	shownum := func(a []Distribution) int {
		x := 0
		e := 1
		for _, v := range a {
			if v.Mean > 0 {
				x += e
			}
			e *= 2
		}
		fmt.Println(x)
		return x
	}

	g := 0.0
	g1 := make([]Distribution, n)
	type Particle struct {
		X []Distribution
		P []Distribution
		F float64
		V []float64
	}
	length := rnd.Intn(4) + 1
	particles := make([]Particle, length)
	pair := func() []int {
		a := make([]int, Width)
		for i := range a {
			a[i] = rnd.Intn(length)
		}
		return a
	}
	samples := 1024
	sample := func(a []int) (avg, sd float64) {
		i := 0
		for i < samples {
			cost := uint64(0)
			for _, value := range a {
				x := uint64(0)
				e := uint64(1)
				for _, v := range particles[value].X {
					if (rnd.NormFloat64()+v.Mean)*v.StdDev > 0 {
						x += e
					}
					e *= 2
				}
				xx := uint64(0)
				if x > 0 {
					xx = uint64(target) % x
				}
				cost += xx
			}
			avg += float64(cost)
			sd += float64(cost) * float64(cost)
			i += 1
		}
		avg /= float64(samples)
		sd = math.Sqrt(sd/float64(samples) - avg*avg)
		return avg, sd
	}
	for i := range particles {
		a := make([]Distribution, 0, n)
		for i := 0; i < n; i++ {
			a = append(a, Distribution{Mean: (2*rnd.Float64() - 1) * 3, StdDev: 1})
		}
		particles[i].X = a
		x := make([]Distribution, n)
		copy(x, a)
		particles[i].P = x
		v := make([]float64, n)
		for i := range v {
			v[i] = (2*rnd.Float64() - 1) * 3
		}
		particles[i].V = v
	}

	graph := pagerank.NewGraph64()
	values := make([]float64, length)
	for range particles {
		set := pair()
		a, _ := sample(set)
		values[set[0]] = a
		values[set[1]] = a
		graph.Link(uint64(set[0]), uint64(set[1]), 2*float64(target)-a)
		graph.Link(uint64(set[1]), uint64(set[0]), 2*float64(target)-a)
	}
	exit := false
	status := false
	graph.Rank(0.85, 0.000001, func(node uint64, rank float64) {
		if exit {
			return
		}
		if rank >= g {
			g = rank
			fmt.Println(g)
			particles[node].F = rank
			copy(g1, particles[node].P)
			x := shownum(g1)
			if target%x == 0 {
				if x == 1 || x == target {
					exit = true
					status = false
				} else {
					fmt.Println(target / x)
					exit = true
					status = true
				}
			}
		}
	})
	if exit {
		return status
	}

	w, w1, w2 := rnd.Float64(), rnd.Float64(), rnd.Float64()
	for j := 0; j < rnd.Intn(16)+1; j++ {
		for i := range particles {
			for j := range particles[i].X {
				rp, rg := rnd.Float64(), rnd.Float64()
				particles[i].V[j] = w*particles[i].V[j] +
					w1*rp*(particles[i].P[j].Mean-particles[i].X[j].Mean) +
					w2*rg*(g1[j].Mean-particles[i].X[j].Mean)
			}
			for j := range particles[i].X {
				particles[i].X[j].Mean += particles[i].V[j]
			}
		}
		graph := pagerank.NewGraph64()
		for range particles {
			set := pair()
			a, _ := sample(set)
			values[set[0]] = a
			values[set[1]] = a
			graph.Link(uint64(set[0]), uint64(set[1]), 2*float64(target)-a)
			graph.Link(uint64(set[1]), uint64(set[0]), 2*float64(target)-a)
		}
		exit := false
		status := false
		graph.Rank(0.85, 0.000001, func(node uint64, rank float64) {
			if exit {
				return
			}
			if rank >= particles[node].F {
				particles[node].F = rank
				copy(particles[node].P, particles[node].X)
				if rank >= g {
					g = rank
					fmt.Println(g)
					copy(g1, particles[node].P)
					x := shownum(g1)
					if target%x == 0 {
						if x == 1 || x == target {
							exit = true
							status = false
						} else {
							fmt.Println(target / x)
							exit = true
							status = true
						}
					}
				}
			}
		})
		if exit {
			return status
		}
	}
	return false
}

func Numbers() {
	rnd := rand.New(rand.NewSource(1))
	type Distribution struct {
		Mean   float64
		StdDev float64
	}
	target := *FlagTarget
	n := int(math.Ceil(math.Log2(float64(target))))
	a := Distribution{Mean: 0, StdDev: math.Ceil(math.Sqrt(float64(target)))}
	b := Distribution{Mean: 0, StdDev: math.Ceil(math.Sqrt(float64(target)))}
	fmt.Println(n)
	fmt.Println(a)
	fmt.Println(b)
	shownum := func(a Distribution) {
		fmt.Println(a.Mean)
	}
	samples := 8 * 1024
	sample := func(a, b Distribution, ia, ib int, d float64) (avg, sd float64) {
		i := 0
		for i < samples {
			x := int((rnd.NormFloat64() + a.Mean) * a.StdDev)
			y := int((rnd.NormFloat64() + b.Mean) * b.StdDev)
			xx := 0
			if x > 0 {
				xx = target % x
			}
			yy := 0
			if y > 0 {
				yy = target % y
			}
			cost := target - x*y
			if cost < 0 {
				cost = -cost
			}
			cost += yy + xx
			avg += float64(cost)
			sd += float64(cost) * float64(cost)
			i += 1
		}
		avg /= float64(samples)
		sd = math.Sqrt(sd/float64(samples) - avg*avg)
		return avg, sd
	}

	avg, sd := sample(a, b, -1, -1, 0)
	j := 0
	for j < 1000 {
		if sd == 0 {
			break
		}
		type Sample struct {
			DeltasA Distribution
			DeltasB Distribution
			Avg     float64
			Sd      float64
		}
		samples := make([]Sample, 16)
		for i := range samples {
			samples[i].DeltasA = Distribution{Mean: a.Mean + rnd.NormFloat64(), StdDev: a.StdDev + rnd.NormFloat64()}
			if samples[i].DeltasA.StdDev < 0 {
				samples[i].DeltasA.StdDev = -samples[i].DeltasA.StdDev
			}
			samples[i].DeltasB = Distribution{Mean: b.Mean + rnd.NormFloat64(), StdDev: b.StdDev + rnd.NormFloat64()}
			if samples[i].DeltasB.StdDev < 0 {
				samples[i].DeltasB.StdDev = -samples[i].DeltasB.StdDev
			}
			samples[i].Avg, samples[i].Sd = sample(samples[i].DeltasA, samples[i].DeltasB, -1, -1, 0)
		}
		sort.Slice(samples, func(i, j int) bool {
			return samples[i].Avg < samples[j].Avg
		})
		avg = samples[0].Avg
		sd = samples[0].Sd
		a = samples[0].DeltasA
		b = samples[0].DeltasB
		fmt.Println(j, avg, sd)
		j++
	}
	fmt.Println(a)
	fmt.Println(b)
	shownum(a)
	shownum(b)
}
