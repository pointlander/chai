// Copyright 2023 The Chai Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"math/big"
	"math/rand"
	"sort"

	"github.com/pointlander/pagerank"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

var (
	// FlagNewton newton's method
	FlagNewton = flag.Bool("newton", false, "newton's method")
	// FlagGraphical graphical mode
	FlagGraphical = flag.Bool("graphical", false, "graphical mode")
	// FlagGradient bits mode
	FlagGradient = flag.Bool("gradient", false, "gradient mode")
	// FlagNumbers numbers mode
	FlagNumbers = flag.Bool("numbers", false, "numbers mode")
	// FlagSwarm swarm mode
	FlagSwarm = flag.Bool("swarm", false, "swarm mode")
	// FlagTarget is the target value
	FlagTarget = flag.Int("target", 77, "target value")
)

// Norm is the normal distribution
func Norm(x, mean, std float64) float64 {
	x -= mean
	return math.Exp(-(x*x)/(2*std*std)) / (std * math.Sqrt(2*math.Pi))
}

// DNorm is the derivative of the normal distribution
func DNorm(x, mean, std float64) float64 {
	x -= mean
	return -x * math.Exp(-(x*x)/(2*std*std)) / (std * std * std * math.Sqrt(2*math.Pi))
}

func main() {
	flag.Parse()

	if *FlagNewton {
		Newton()
		return
	}

	if *FlagGraphical {
		for seed := 1; seed < 1000; seed++ {
			if Graphical(seed) {
				return
			}
		}
		return
	}

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
		for seed := 1; seed != 0; seed++ {
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

	j := math.MaxUint32 >> 16
	primes := make([]int, 2)
	for i := 0; i < 2; i++ {
		for {
			if big.NewInt(int64(j)).ProbablyPrime(100) {
				primes[i] = j
				j--
				break
			}
			j--
		}
	}
	fmt.Println(primes, primes[0]*primes[1])
}

// Newton implements newton's method for factoring numbers
func Newton() {
	rnd := rand.New(rand.NewSource(1))

	type Distribution struct {
		Mean   float64
		StdDev float64
	}
	target := *FlagTarget
	n := int(math.Ceil(math.Log2(math.Sqrt(float64(target)))))
	a := make([][]Distribution, 32)
	aa := make([][]Distribution, len(a))
	for j := range a {
		a[j] = make([]Distribution, 0, n)
		for i := 0; i < n; i++ {
			a[j] = append(a[j], Distribution{Mean: rnd.NormFloat64(), StdDev: 1})
		}
		aa[j] = make([]Distribution, len(a[j]))
		copy(aa[j], a[j])
	}

	size := int(math.Ceil(math.Log2(float64(target))))
	t := make([]Distribution, 0, size)
	for i := 0; i < size; i++ {
		t = append(t, Distribution{Mean: rnd.NormFloat64(), StdDev: 1})
	}

	samples := 8 * 1024
	sample := func(t []Distribution, a [][]Distribution) (d []float64, avg, sd float64) {
		for i := 0; i < samples; i++ {
			cost := 0.0
			for _, a := range a {
				x := uint64(0)
				e := uint64(1)
				for _, v := range a {
					if (rnd.NormFloat64()+v.Mean)*v.StdDev > 0 {
						x += e
					}
					e *= 2
				}
				tt := uint64(0)
				e = 1
				for _, v := range t {
					if (rnd.NormFloat64()+v.Mean)*v.StdDev > 0 {
						tt += e
					}
					e *= 2
				}
				xx := uint64(0)
				if x > 0 {
					xx = tt % x
				}
				cost += float64(xx) + math.Abs(float64(target)-float64(tt))/float64(target)
			}
			d = append(d, cost)
			avg += cost
			sd += cost * cost
		}
		avg /= float64(samples * len(a))
		sd = math.Sqrt(sd/float64(samples*len(a)) - avg*avg)
		return d, avg, sd
	}
	d, avg, sd := sample(t, a)
	fmt.Println(avg, sd)

	shownum := func(a []Distribution) int {
		x := 0
		e := 1
		for _, v := range a {
			if v.Mean > 0 {
				x += e
			}
			e *= 2
		}
		return x
	}

	min := math.MaxFloat64
Search:
	for e := 0; true; e++ {
		_, avg, sd := sample(t, a)
		if avg < min {
			min = avg
			for j := range a {
				copy(aa[j], a[j])
			}
		} else {
			for j := range a {
				copy(a[j], aa[j])
			}
		}
		fmt.Println(e, min)
		for i := range a {
			factor := rnd.Float64()
			for j := range t {
				if rnd.Intn(2) == 0 {
					t[j].Mean -= factor * Norm(t[j].Mean, avg, sd) / DNorm(t[j].Mean, avg, sd)
				} else {
					t[j].Mean += factor * Norm(t[j].Mean, avg, sd) / DNorm(t[j].Mean, avg, sd)
				}
			}
			for j := range a[i] {
				if rnd.Intn(2) == 0 {
					a[i][j].Mean -= factor * Norm(a[i][j].Mean, avg, sd) / DNorm(a[i][j].Mean, avg, sd)
				} else {
					a[i][j].Mean += factor * Norm(a[i][j].Mean, avg, sd) / DNorm(a[i][j].Mean, avg, sd)
				}
			}
		}
		for i := range a {
			x := shownum(a[i])
			if x == 0 {
				continue
			}
			if target%x == 0 {
				if x == 1 || x == target {
					continue
				} else {
					fmt.Println(x, target/x)
					break Search
				}
			}
		}
	}

	values := make(plotter.Values, 0, 8)
	for _, v := range d {
		values = append(values, v)
	}

	p := plot.New()
	p.Title.Text = "binary"

	histogram, err := plotter.NewHist(values, 20)
	if err != nil {
		panic(err)
	}
	p.Add(histogram)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "binary.png")
	if err != nil {
		panic(err)
	}
}

// Graphical implements graphical search for factoring numbers
func Graphical(seed int) bool {
	fmt.Println("seed", seed)
	rnd := rand.New(rand.NewSource(int64(seed)))
	type Distribution struct {
		Mean   float64
		StdDev float64
	}
	type Node struct {
		A      []Distribution
		Weight float64
	}
	target := *FlagTarget
	n := int(math.Ceil(math.Log2(float64(target))))
	a := make([]Node, 32)
	for i := range a {
		for j := 0; j < n; j++ {
			a[i].A = append(a[i].A, Distribution{Mean: rnd.NormFloat64(), StdDev: rnd.NormFloat64()})
		}
	}

	samples := 1024
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
			cost := target*target - yy*xx
			avg += float64(cost)
			sd += float64(cost) * float64(cost)
			i += 1
		}
		avg /= float64(samples)
		sd = math.Sqrt(sd/float64(samples) - avg*avg)
		return avg, sd
	}

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

	for e := 0; e < 16; e++ {
		graph := pagerank.NewGraph64()
		for i := range a {
			remainder := a[i:]
			for j := range remainder {
				avg, _ := sample(a[i].A, a[j].A)
				graph.Link(uint64(i), uint64(j), avg)
				graph.Link(uint64(j), uint64(i), avg)
			}
		}
		graph.Rank(0.85, 0.000001, func(node uint64, rank float64) {
			a[node].Weight = rank
		})
		d := make([]Distribution, n)
		for i := range a {
			for j := range d {
				d[j].Mean += a[i].A[j].Mean * a[i].Weight * rnd.Float64()
				d[j].StdDev += a[i].A[j].StdDev * a[i].Weight * rnd.Float64()
			}
		}
		a = append(a, Node{A: d})
		x := shownum(d)
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
	}
	return false
}

// Gradient implements pseudo-gradient descent for factoring numbers
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
	Width = 4
)

// Swarm implements particle swarm optimization for factoring numbers
func Swarm(seed int) bool {
	fmt.Println("seed", seed)
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

	g := math.MaxFloat64
	g1 := make([]Distribution, n)
	type Particle struct {
		X []Distribution
		P []Distribution
		F float64
		V []float64
	}
	length := rnd.Intn(16) + 1
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

	for range particles {
		set := pair()
		a, _ := sample(set)
		if a <= g {
			g = a
			fmt.Println(g)
			d := make([]Distribution, n)
			for _, j := range set {
				particles[j].F = a
				for k := range particles[j].P {
					d[k].Mean += particles[j].P[k].Mean
					d[k].StdDev += particles[j].P[k].StdDev
				}
				x := shownum(particles[j].P)
				if x == 0 {
					return false
				}
				if target%x == 0 {
					if x == 1 || x == target {
						return false
					} else {
						fmt.Println("lucky", target/x)
						return true
					}
				}
			}
			for k := range d {
				d[k].Mean /= float64(len(set))
				d[k].StdDev /= float64(len(set))
			}
			copy(g1, d)
		}
	}

	w, w1, w2 := rnd.Float64(), rnd.Float64(), rnd.Float64()
	for {
		current := g
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
		for range particles {
			set := pair()
			a, _ := sample(set)
			d := make([]Distribution, n)
			s := false
			for _, j := range set {
				if a <= particles[j].F {
					particles[j].F = a
					copy(particles[j].P, particles[j].X)
					if a <= g {
						g = a
						fmt.Println(g)
						s = true
						for k := range particles[j].P {
							d[k].Mean += particles[j].P[k].Mean
							d[k].StdDev += particles[j].P[k].StdDev
						}
						x := shownum(particles[j].P)
						if x == 0 {
							return false
						}
						if target%x == 0 {
							if x == 1 || x == target {
								return false
							} else {
								fmt.Println(target/x, w, w1, w2)
								return true
							}
						}
					}
				}
			}
			if s {
				for k := range d {
					d[k].Mean /= float64(len(set))
					d[k].StdDev /= float64(len(set))
				}
				copy(g1, d)
			}
		}
		if g == current {
			return false
		}
	}
}

// Numbers implements pseudo-gradient descent to find a factor of the target
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
