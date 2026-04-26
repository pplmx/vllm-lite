package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!

// VLLMEngineSpec defines the desired state of VLLMEngine
type VLLMEngineSpec struct {
	// Model name to serve (e.g., "meta-llama/Llama-2-7b-hf")
	// +kubebuilder:validation:Required
	Model string `json:"model"`

	// Number of replicas
	// +optional
	Replicas *int32 `json:"replicas,omitempty"`

	// Container image for vLLM
	// +optional
	Image string `json:"image,omitempty"`

	// GPU resources
	// +optional
	GPUResource GPUResource `json:"gpuResource,omitempty"`

	// Max model length (sequence length)
	// +optional
	MaxModelLen *int32 `json:"maxModelLen,omitempty"`

	// Tensor parallelism degree
	// +optional
	TensorParallelSize *int32 `json:"tensorParallelSize,omitempty"`

	// Additional arguments for vLLM server
	// +optional
	Args []string `json:"args,omitempty"`
}

type GPUResource struct {
	// Number of GPUs
	// +optional
	Count int32 `json:"count,omitempty"`

	// GPU memory in Gi
	// +optional
	Memory string `json:"memory,omitempty"`
}

// VLLMEngineStatus defines the observed state of VLLMEngine
type VLLMEngineStatus struct {
	// Current replicas
	Replicas int32 `json:"replicas,omitempty"`

	// Ready replicas
	ReadyReplicas int32 `json:"readyReplicas,omitempty"`

	// Available replicas
	AvailableReplicas int32 `json:"availableReplicas,omitempty"`

	// Service URL
	ServiceURL string `json:"serviceURL,omitempty"`

	// Conditions
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Model",type=string,JSONPath=`.spec.model`
// +kubebuilder:printcolumn:name="Replicas",type=integer,JSONPath=`.spec.replicas`
// +kubebuilder:printcolumn:name="Ready",type=integer,JSONPath=`.status.readyReplicas`
// +kubebuilder:printcolumn:name="URL",type=string,JSONPath=`.status.serviceURL`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// VLLMEngine is the Schema for the vllmengines API
type VLLMEngine struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   VLLMEngineSpec   `json:"spec,omitempty"`
	Status VLLMEngineStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// VLLMEngineList contains a list of VLLMEngine
type VLLMEngineList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []VLLMEngine `json:"items"`
}

func init() {
	SchemeBuilder.Register(&VLLMEngine{}, &VLLMEngineList{})
}
