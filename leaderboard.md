# SpatialBench Main Leaderboard

Metrics are reported across Single Frame, Sparse, Medium, Dense, and Average settings. `Time` is the per-sequence inference time in seconds in the sparse regime. Lower is better for AbsRel and ATE; higher is better for AUC@30 and F-Score.

Bold values mark the best result within each method group. Superscripts `1`, `2`, and `3` mark the overall top-three values highlighted in the original LaTeX table. Average values in parentheses are computed without the dense regime because the method OOMs or times out there.

<div style="overflow-x: auto; width: 100%;">

<table style="width: max-content; min-width: 1600px;">
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">#Params (M)</th>
<th rowspan="2">Time (s)</th>
<th colspan="1">Single Frame</th>
<th colspan="2">Sparse</th>
<th colspan="4">Medium</th>
<th colspan="4">Dense</th>
<th colspan="4">Average</th>
</tr>
<tr>
<th>AbsRel &darr;</th>
<th>AbsRel &darr;</th>
<th>AUC@30 &uarr;</th>
<th>AbsRel &darr;</th>
<th>AUC@30 &uarr;</th>
<th>ATE &darr;</th>
<th>F-Score &uarr;</th>
<th>AbsRel &darr;</th>
<th>AUC@30 &uarr;</th>
<th>ATE &darr;</th>
<th>F-Score &uarr;</th>
<th>AbsRel &darr;</th>
<th>AUC@30 &uarr;</th>
<th>ATE &darr;</th>
<th>F-Score &uarr;</th>
</tr>
</thead>
<tbody>
<tr><th colspan="18">Optimization-based</th></tr>
<tr><td>DUSt3R</td><td>571.17</td><td><strong>7.59</strong></td><td><strong>0.385</strong></td><td>0.257</td><td>0.498</td><td>0.276</td><td>0.448</td><td><strong>1.691</strong></td><td>0.343</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>(0.267)</td><td>(0.473)</td><td>(1.691)</td><td>(0.343)</td></tr>
<tr><td>MASt3R</td><td>688.64</td><td>8.17</td><td>0.456</td><td><strong>0.209</strong></td><td><strong>0.568</strong></td><td><strong>0.259</strong></td><td><strong>0.522</strong></td><td>1.911</td><td><strong>0.370</strong></td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>(0.234)</td><td>(0.545)</td><td>(1.911)</td><td>(0.370)</td></tr>

<tr><th colspan="18">End-to-End Feed-Forward</th></tr>
<tr><td>VGGT</td><td>1256.54</td><td>0.40</td><td>0.184<sup>2</sup></td><td>0.105</td><td>0.700</td><td>0.125</td><td>0.687</td><td>0.727</td><td>0.661</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>(0.115)</td><td>(0.693)</td><td>(0.727)</td><td>(0.661)</td></tr>
<tr><td>Fast3R</td><td>647.55</td><td>0.90</td><td>0.350</td><td>0.260</td><td>0.392</td><td>0.255</td><td>0.386</td><td>6.582</td><td>0.300</td><td>0.331</td><td>0.232</td><td><strong>13.68</strong></td><td>0.224</td><td>0.282</td><td>0.337</td><td>10.13</td><td>0.262</td></tr>
<tr><td>FastVGGT</td><td>1157.94</td><td>0.24</td><td><strong>0.183</strong><sup>1</sup></td><td>0.113</td><td>0.631</td><td>0.105</td><td>0.662</td><td>0.738</td><td>0.576</td><td>0.120<sup>2</sup></td><td><strong>0.588</strong></td><td>19.23</td><td><strong>0.479</strong><sup>3</sup></td><td>0.113<sup>3</sup></td><td>0.627</td><td>9.985</td><td><strong>0.528</strong></td></tr>
<tr><td>MUSt3R</td><td>423.43</td><td>0.96</td><td>0.429</td><td>0.165</td><td>0.614</td><td>0.162</td><td>0.643</td><td>3.097</td><td>0.507</td><td>T.O</td><td>T.O</td><td>T.O</td><td>T.O</td><td>(0.164)</td><td>(0.629)</td><td>(3.097)</td><td>(0.507)</td></tr>
<tr><td>MapAnything</td><td>1228.49</td><td>0.22</td><td>0.451</td><td>0.153</td><td>0.579</td><td>0.146</td><td>0.579</td><td>1.737</td><td>0.420</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>(0.150)</td><td>(0.579)</td><td>(1.737)</td><td>(0.420)</td></tr>
<tr><td>OmniVGGT</td><td>1217.49</td><td>0.22<sup>3</sup></td><td>0.188<sup>3</sup></td><td>0.117</td><td>0.665</td><td>0.111</td><td>0.665</td><td>1.491</td><td>0.595</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>(0.114)</td><td>(0.665)</td><td>(1.491)</td><td>(0.595)</td></tr>
<tr><td>&pi;<sup>3</sup></td><td>958.70</td><td><strong>0.20</strong><sup>2</sup></td><td>0.478</td><td>0.092</td><td>0.742</td><td>0.082<sup>3</sup></td><td>0.749</td><td>0.565</td><td>0.649</td><td><strong>0.109</strong><sup>1</sup></td><td>0.524</td><td>16.39</td><td>0.332</td><td><strong>0.094</strong><sup>1</sup></td><td><strong>0.672</strong><sup>3</sup></td><td><strong>8.480</strong></td><td>0.490</td></tr>
<tr><td>&pi;<sup>3</sup>-X</td><td>1360.03</td><td>0.24</td><td>0.371</td><td>0.084<sup>3</sup></td><td>0.741</td><td>0.078<sup>2</sup></td><td>0.744</td><td><strong>0.369</strong><sup>1</sup></td><td>0.658</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>(0.081)</td><td>(0.742)</td><td>(0.369)</td><td>(0.658)</td></tr>
<tr><td>AMB3R</td><td>1563.12</td><td>0.53</td><td>0.466</td><td>0.088</td><td>0.739</td><td>0.085</td><td>0.727</td><td>0.645</td><td>0.554</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>(0.087)</td><td>(0.733)</td><td>(0.645)</td><td>(0.554)</td></tr>
<tr><td>DA3-Small</td><td>34.30</td><td>0.39</td><td>0.385</td><td>0.191</td><td>0.476</td><td>0.176</td><td>0.479</td><td>4.850</td><td>0.432</td><td>0.208</td><td>0.368</td><td>28.12</td><td>0.325</td><td>0.192</td><td>0.441</td><td>16.48</td><td>0.379</td></tr>
<tr><td>DA3-Base</td><td>135.37</td><td>0.40</td><td>0.349</td><td>0.159</td><td>0.566</td><td>0.142</td><td>0.562</td><td>3.865</td><td>0.515</td><td>0.166</td><td>0.436</td><td>26.35</td><td>0.399</td><td>0.156</td><td>0.521</td><td>15.11</td><td>0.457</td></tr>
<tr><td>DA3-Large</td><td>410.94</td><td>0.41</td><td>0.333</td><td>0.128</td><td>0.688</td><td>0.105</td><td>0.701</td><td>2.722</td><td>0.626</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>(0.116)</td><td>(0.694)</td><td>(2.722)</td><td>(0.626)</td></tr>
<tr><td>DA3-Giant</td><td>1355.67</td><td>0.47</td><td>0.368</td><td>0.095</td><td>0.785<sup>3</sup></td><td>0.086</td><td>0.776<sup>2</sup></td><td>1.161</td><td><strong>0.742</strong><sup>1</sup></td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>(0.091)</td><td>(0.780)</td><td>(1.161)</td><td>(0.742)</td></tr>
<tr><td>DA3-Nested</td><td>1689.85</td><td>0.52</td><td>0.364</td><td>0.106</td><td>0.779</td><td>0.086</td><td>0.770<sup>3</sup></td><td>1.980</td><td>0.737<sup>2</sup></td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>(0.096)</td><td>(0.774)</td><td>(1.980)</td><td>(0.737)</td></tr>
<tr><td>WorldMirror</td><td>1263.34</td><td>0.22</td><td>0.349</td><td>0.139</td><td>0.660</td><td>0.118</td><td>0.674</td><td>1.357</td><td>0.575</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>(0.129)</td><td>(0.667)</td><td>(1.357)</td><td>(0.575)</td></tr>
<tr><td>VGGT-Omega</td><td>1143.81</td><td>0.48</td><td>0.516</td><td><strong>0.077</strong><sup>1</sup></td><td><strong>0.803</strong><sup>1</sup></td><td><strong>0.067</strong><sup>1</sup></td><td><strong>0.795</strong><sup>1</sup></td><td>0.659</td><td>0.706</td><td>OOM<sup>*</sup></td><td>OOM<sup>*</sup></td><td>OOM<sup>*</sup></td><td>OOM<sup>*</sup></td><td>(0.072)</td><td>(0.799)</td><td>(0.659)</td><td>(0.706)</td></tr>

<tr><th colspan="18">Online</th></tr>
<tr><td>Spann3r<sup>224</sup></td><td>658.69</td><td>0.55</td><td>0.370</td><td>0.274</td><td>0.329</td><td>0.252</td><td>0.361</td><td>4.312</td><td>0.254</td><td>0.315</td><td>0.246</td><td>26.48</td><td>0.159</td><td>0.280</td><td>0.312</td><td>15.39</td><td>0.207</td></tr>
<tr><td>CUT3R</td><td>793.31</td><td>0.41</td><td>0.247</td><td>0.196</td><td>0.519</td><td>0.189</td><td>0.469</td><td>2.676</td><td>0.286</td><td>0.260</td><td>0.165</td><td>25.54</td><td>0.109</td><td>0.215</td><td>0.384</td><td>14.11</td><td>0.198</td></tr>
<tr><td>MonST3R</td><td>571.17</td><td>20.81</td><td>0.309</td><td>0.227</td><td>0.269</td><td>0.241</td><td>0.195</td><td>2.234</td><td>0.081</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>(0.234)</td><td>(0.232)</td><td>(2.234)</td><td>(0.081)</td></tr>
<tr><td>Point3R</td><td>828.01</td><td>1.05</td><td>0.379</td><td>0.221</td><td>0.339</td><td>0.228</td><td>0.303</td><td>6.512</td><td>0.211</td><td>0.285</td><td>0.212</td><td>28.09</td><td>0.139</td><td>0.244</td><td>0.285</td><td>17.3</td><td>0.175</td></tr>
<tr><td>Stream3R-S</td><td>1190.60</td><td>0.62</td><td>0.409</td><td>0.114</td><td>0.603</td><td>0.204</td><td>0.427</td><td>5.717</td><td>0.348</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>(0.159)</td><td>(0.515)</td><td>(5.717)</td><td>(0.348)</td></tr>
<tr><td>Stream3R-W</td><td>1190.60</td><td>0.62</td><td>0.409</td><td>0.117</td><td>0.597</td><td>0.240</td><td>0.364</td><td>6.756</td><td>0.323</td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>(0.178)</td><td>(0.481)</td><td>(6.756)</td><td>(0.323)</td></tr>
<tr><td>StreamVGGT</td><td>1256.54</td><td>0.85</td><td>0.219</td><td>0.154</td><td>0.598</td><td>0.171</td><td>0.562</td><td>4.940</td><td>0.397</td><td>0.198</td><td>0.413</td><td>26.9</td><td>0.251</td><td>0.174</td><td>0.524</td><td>15.92</td><td>0.324</td></tr>
<tr><td>Page4D</td><td>1256.81</td><td>0.56</td><td>0.228</td><td><strong>0.112</strong></td><td>0.608</td><td><strong>0.107</strong></td><td>0.618</td><td>0.855</td><td><strong>0.423</strong></td><td>OOM</td><td>OOM</td><td>OOM</td><td>OOM</td><td>(0.110)</td><td>(0.613)</td><td>(0.855)</td><td>(0.423)</td></tr>
<tr><td>InfiniteVGGT</td><td>1256.54</td><td>0.46</td><td><strong>0.217</strong></td><td>0.154</td><td>0.596</td><td>0.170</td><td>0.563</td><td>4.964</td><td>0.402</td><td>0.197</td><td>0.416</td><td>27.01</td><td>0.254</td><td>0.174</td><td>0.525</td><td>15.99</td><td>0.328</td></tr>
<tr><td>Wint3R</td><td>749.46</td><td>0.41</td><td>0.619</td><td>0.157</td><td>0.499</td><td>0.144</td><td>0.444</td><td>3.944</td><td>0.401</td><td>0.234</td><td>0.202</td><td>27.8</td><td>0.114</td><td>0.178</td><td>0.382</td><td>15.87</td><td>0.258</td></tr>
<tr><td>LongStream-B</td><td>1190.60</td><td>0.59</td><td>0.523</td><td>0.153</td><td>0.549</td><td>0.224</td><td>0.455</td><td>0.925</td><td>0.135</td><td>0.269</td><td>0.294</td><td>5.766</td><td>0.083</td><td>0.215</td><td>0.432</td><td>3.346</td><td>0.109</td></tr>
<tr><td>LongStream-S</td><td>1190.60</td><td>0.83</td><td>0.523</td><td>0.151</td><td>0.543</td><td>0.166</td><td>0.385</td><td>1.188</td><td>0.126</td><td>0.279</td><td>0.218</td><td>10.08</td><td>0.083</td><td>0.199</td><td>0.382</td><td>5.632</td><td>0.105</td></tr>
<tr><td>LingbotMap<sup>*</sup>-W</td><td>1157.94</td><td><strong>0.30</strong></td><td>0.333</td><td>0.138</td><td><strong>0.650</strong></td><td>0.114</td><td>0.641</td><td>0.509</td><td>0.362</td><td>0.167</td><td>0.553</td><td>4.694</td><td>0.352</td><td>0.139</td><td>0.615</td><td>2.601</td><td>0.357</td></tr>
<tr><td>LingbotMap<sup>*</sup>-S</td><td>1157.94</td><td>0.33</td><td>0.333</td><td>0.138</td><td><strong>0.650</strong></td><td>0.114</td><td><strong>0.647</strong></td><td><strong>0.508</strong></td><td>0.411</td><td><strong>0.139</strong><sup>3</sup></td><td><strong>0.627</strong><sup>1</sup></td><td><strong>3.470</strong><sup>2</sup></td><td><strong>0.472</strong></td><td><strong>0.130</strong></td><td><strong>0.641</strong></td><td><strong>1.989</strong><sup>2</sup></td><td><strong>0.441</strong></td></tr>

<tr><th colspan="18">Chunk-wise</th></tr>
<tr><td>VGGT-Long</td><td>1256.54</td><td><strong>0.20</strong><sup>1</sup></td><td><strong>0.184</strong><sup>2</sup></td><td>0.105</td><td>0.700</td><td>0.131</td><td>0.679</td><td>0.512</td><td>0.633</td><td>0.222</td><td>0.507</td><td>8.467</td><td>0.467</td><td>0.152</td><td>0.629</td><td>4.489</td><td>0.550<sup>3</sup></td></tr>
<tr><td>&pi;<sup>3</sup>-Long</td><td>958.70</td><td>0.23</td><td>0.478</td><td><strong>0.092</strong></td><td>0.742</td><td>0.097</td><td>0.740</td><td><strong>0.465</strong><sup>3</sup></td><td>0.590</td><td><strong>0.216</strong></td><td><strong>0.614</strong><sup>2</sup></td><td><strong>4.021</strong><sup>3</sup></td><td>0.251</td><td><strong>0.135</strong></td><td><strong>0.699</strong><sup>1</sup></td><td><strong>2.243</strong><sup>3</sup></td><td>0.421</td></tr>
<tr><td>DA3-Streaming</td><td>1355.67</td><td>0.51</td><td>0.368</td><td>0.095</td><td><strong>0.785</strong><sup>2</sup></td><td><strong>0.091</strong></td><td><strong>0.767</strong></td><td>0.563</td><td><strong>0.725</strong><sup>3</sup></td><td>0.245</td><td>0.546</td><td>8.575</td><td><strong>0.516</strong><sup>1</sup></td><td>0.144</td><td><strong>0.699</strong><sup>1</sup></td><td>4.569</td><td><strong>0.621</strong><sup>1</sup></td></tr>

<tr><th colspan="18">SLAM-based</th></tr>
<tr><td>MASt3R-SLAM</td><td>688.64</td><td>3.04</td><td>0.348</td><td>0.336</td><td>0.190</td><td>0.348</td><td>0.262</td><td>6.075</td><td>0.130</td><td>0.404</td><td>0.311</td><td>25.7</td><td>0.121</td><td>0.363</td><td>0.254</td><td>15.89</td><td>0.126</td></tr>
<tr><td>VGGT-SLAM</td><td>1256.54</td><td><strong>0.57</strong></td><td><strong>0.184</strong><sup>2</sup></td><td><strong>0.105</strong></td><td><strong>0.700</strong></td><td><strong>0.129</strong></td><td><strong>0.645</strong></td><td><strong>0.686</strong></td><td><strong>0.610</strong></td><td><strong>0.211</strong></td><td><strong>0.441</strong></td><td><strong>9.069</strong></td><td><strong>0.384</strong></td><td><strong>0.148</strong></td><td><strong>0.595</strong></td><td><strong>4.877</strong></td><td><strong>0.497</strong></td></tr>

<tr><th colspan="18">Test-Time Training</th></tr>
<tr><td>TTT3R</td><td>793.31</td><td>0.61</td><td>0.247</td><td>0.202</td><td>0.469</td><td>0.179</td><td>0.493</td><td>2.343</td><td>0.294</td><td>0.222</td><td>0.321</td><td>21.07</td><td>0.173</td><td>0.201</td><td>0.428</td><td>11.7</td><td>0.233</td></tr>
<tr><td>Scal3R</td><td>1266.14</td><td>2.32</td><td>0.227</td><td>0.114</td><td><strong>0.732</strong></td><td>0.147</td><td>0.670</td><td><strong>0.400</strong><sup>2</sup></td><td><strong>0.671</strong></td><td>0.244</td><td>0.480</td><td><strong>2.396</strong><sup>1</sup></td><td><strong>0.498</strong><sup>2</sup></td><td>0.168</td><td>0.627</td><td><strong>1.398</strong><sup>1</sup></td><td><strong>0.585</strong><sup>2</sup></td></tr>
<tr><td>LoGeR</td><td>1254.62</td><td><strong>0.26</strong></td><td>0.251</td><td>0.095</td><td>0.687</td><td>0.113</td><td>0.693</td><td>0.591</td><td>0.504</td><td>0.197</td><td>0.552</td><td>5.217</td><td>0.335</td><td>0.135</td><td>0.644</td><td>2.904</td><td>0.419</td></tr>
<tr><td>LoGeR<sup>*</sup></td><td>1254.60</td><td>0.30</td><td><strong>0.200</strong></td><td><strong>0.077</strong><sup>2</sup></td><td>0.708</td><td><strong>0.083</strong></td><td><strong>0.714</strong></td><td>0.566</td><td>0.574</td><td><strong>0.156</strong></td><td><strong>0.598</strong><sup>3</sup></td><td>4.598</td><td>0.421</td><td><strong>0.105</strong><sup>2</sup></td><td><strong>0.673</strong><sup>2</sup></td><td>2.582</td><td>0.497</td></tr>
<tr><td>ZipMap</td><td>1366.87</td><td>--</td><td>0.230</td><td>0.099</td><td>0.657</td><td>0.093</td><td>0.668</td><td>1.186</td><td>0.605</td><td>OOM<sup>*</sup></td><td>OOM<sup>*</sup></td><td>OOM<sup>*</sup></td><td>OOM<sup>*</sup></td><td>(0.096)</td><td>(0.662)</td><td>(1.186)</td><td>(0.605)</td></tr>
<tr><td>VGG-TTT</td><td>1191.99</td><td>--</td><td>0.208</td><td>0.115</td><td>0.542</td><td>0.114</td><td>0.530</td><td>3.127</td><td>0.416</td><td>OOM<sup>*</sup></td><td>OOM<sup>*</sup></td><td>OOM<sup>*</sup></td><td>OOM<sup>*</sup></td><td>(0.114)</td><td>(0.536)</td><td>(3.127)</td><td>(0.416)</td></tr>
</tbody>
</table>

</div>

Notes:

- `S` denotes stream, `B` denotes batch, and `W` denotes window.
- LingbotMap<sup>*</sup> indicates the best checkpoint is selected in each regime.
- OOM means out of memory with memory greater than 140G. T.O means timeout with runtime greater than 4h per scene.
- OOM<sup>*</sup> means out of memory on an A100 80G GPU.
