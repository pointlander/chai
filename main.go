// Copyright 2023 The Chai Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
)

func main() {
	rnd := rand.New(rand.NewSource(1))
	target := 77
	n := int(math.Ceil(math.Log2(float64(target))))
	step := .1
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
	costs := []int{}
	samples := 1024
	sample := func(a, b []float64, ia, ib int, d float64) (avg, ss float64) {
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
			if d == 0 {
				costs = append(costs, cost)
			}
			avg += float64(cost)
			ss += float64(cost) * float64(cost)
			i += 1
		}
		avg /= float64(samples)
		ss = math.Sqrt(ss/float64(samples) - avg*avg)
		return
	}
	avg1, ss1 := sample(a, b, -1, -1, 0)
	avg2, ss2 := avg1, ss1
	j := 0
	for j < 1000 {
		if (ss1 == 0) && (ss2 == 0) {
			break
		}
		min := 0.0
		index := -1
		for i := 0; i < n; i++ {
			step = rnd.NormFloat64()
			x, sx := sample(a, b, i, -1, step)
			y, sy := sample(a, b, i, -1, -step)
			if x < avg1 {
				avg1 = x
				ss1 = sx
				min = step
				index = i
			}
			if y < avg1 {
				avg1 = y
				ss1 = sy
				min = -step
				index = i
			}
		}
		if index >= 0 {
			a[index] += min
			fmt.Println(j, avg1, ss1)
		}
		min = 0.0
		index = -1
		for i := 0; i < n; i++ {
			step = rnd.NormFloat64()
			x, sx := sample(a, b, -1, i, step)
			y, sy := sample(a, b, -1, i, -step)
			if x < avg2 {
				avg2 = x
				ss2 = sx
				min = step
				index = i
			}
			if y < avg2 {
				avg2 = y
				ss2 = sy
				min = -step
				index = i
			}
		}
		if index >= 0 {
			b[index] += min
			fmt.Println(j, avg2, ss2)
		}
		j++
	}
	fmt.Println(a)
	fmt.Println(b)
	shownum(a)
	shownum(b)
}
