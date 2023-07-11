// Copyright 2023 The Chai Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

func main() {
	rnd := rand.New(rand.NewSource(1))
	target := 77
	n := int(math.Ceil(math.Log2(float64(target))))
	a := make([]float64, 0, n)
	b := make([]float64, 0, n)
	for i := 0; i < n; i++ {
		a = append(a, 0)
		b = append(b, 0)
	}
	fmt.Println(n)
	fmt.Println(a)
	fmt.Println(b)
	shownum := func(a []float64) {
		x := 0.0
		e := 1.0
		for _, v := range a {
			if v > 0 {
				x += e
			}
			e *= 2
		}
		fmt.Println(x)
	}
	samples := 8 * 1024
	sample := func(a, b []float64, ia, ib int, d float64) (avg, sd float64) {
		i := 0
		for i < samples {
			x := 0
			y := 0
			e := 1
			k := 0
			for _, v := range a {
				if k == ia {
					v = v + d
				}
				if (rnd.NormFloat64() + v) > 0 {
					x += e
				}
				e *= 2
				k++
			}
			e = 1
			k = 0
			for _, v := range b {
				if k == ib {
					v = v + d
				}
				if (rnd.NormFloat64() + v) > 0 {
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
			DeltasA []float64
			DeltasB []float64
			Avg     float64
			Sd      float64
		}
		samples := make([]Sample, 16)
		for i := range samples {
			samples[i].DeltasA = make([]float64, len(a))
			copy(samples[i].DeltasA, a)
			for j := range samples[i].DeltasA {
				samples[i].DeltasA[j] += rnd.NormFloat64()
			}
			samples[i].DeltasB = make([]float64, len(b))
			copy(samples[i].DeltasB, b)
			for j := range samples[i].DeltasB {
				samples[i].DeltasB[j] += rnd.NormFloat64()
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
