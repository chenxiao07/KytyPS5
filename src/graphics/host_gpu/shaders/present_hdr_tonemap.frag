#version 450

// The presenter's copy of a guest HDR10 frame (BT.2100 PQ) into the HDR10 swapchain image, its highlights fitted to the
// display's peak: the BT.2390 EETF (the roll-off TVs apply to content mastered brighter than they are), from the peak
// the game's frames are taken to reach (--hdr-source-nits) to the display's (HdrOutputLuminance). Below the knee
// (about 145 nits for 10,000 into 600) every pixel stays as the game made it; above it the highlights are compressed
// smoothly instead of clipped by the display. Applied to each pixel's largest channel, the others scaled with it: hue
// and saturation kept. The same scaling as the blit it replaces (a linear filter).

layout(set = 0, binding = 0) uniform sampler2D source_image;

layout(push_constant) uniform Constants {
	vec2  offset;         // the destination rectangle's top left, in pixels
	vec2  scale;          // source texture coordinates per destination pixel
	float source_peak_pq; // the frames' peak, PQ encoded (1.0: 10,000 nits)
	float target_peak_pq; // the display's peak, PQ encoded (below source_peak_pq)
} constants;

layout(location = 0) out vec4 color;

const float M1 = 2610.0 / 16384.0;
const float M2 = 2523.0 / 32.0;
const float C1 = 3424.0 / 4096.0;
const float C2 = 2413.0 / 128.0;
const float C3 = 2392.0 / 128.0;

// PQ signal to linear light, 1.0 = 10,000 nits; and back.
vec3 PqToLinear(vec3 e) {
	vec3 p = pow(clamp(e, 0.0, 1.0), vec3(1.0 / M2));
	return pow(max(p - C1, 0.0) / (C2 - C3 * p), vec3(1.0 / M1));
}

float LinearToPq(float y) {
	float p = pow(clamp(y, 0.0, 1.0), M1);
	return pow((C1 + C2 * p) / (1.0 + C3 * p), M2);
}

float PqToLinear(float e) {
	return PqToLinear(vec3(e)).x;
}

// BT.2390's EETF on a PQ value (the mastering and display black levels taken as 0).
float Eetf(float e) {
	float e1      = min(e / constants.source_peak_pq, 1.0);
	float max_lum = constants.target_peak_pq / constants.source_peak_pq;
	float ks      = max(1.5 * max_lum - 0.5, 0.0);
	if (e1 > ks) {
		float t  = (e1 - ks) / (1.0 - ks);
		float t2 = t * t;
		float t3 = t2 * t;
		e1 = (2.0 * t3 - 3.0 * t2 + 1.0) * ks + (t3 - 2.0 * t2 + t) * (1.0 - ks) + (-2.0 * t3 + 3.0 * t2) * max_lum;
	}
	return e1 * constants.source_peak_pq;
}

void main() {
	vec2 uv     = (gl_FragCoord.xy - constants.offset) * constants.scale;
	vec4 pixel  = textureLod(source_image, uv, 0.0);
	vec3 linear = PqToLinear(pixel.rgb);
	float peak  = max(max(linear.r, linear.g), linear.b);
	if (peak > 0.0) {
		linear *= PqToLinear(Eetf(LinearToPq(peak))) / peak;
	}
	color = vec4(LinearToPq(linear.r), LinearToPq(linear.g), LinearToPq(linear.b), pixel.a);
}
