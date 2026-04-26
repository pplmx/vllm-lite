package controller

import (
	"context"
	"fmt"
	"reflect"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	v1alpha1 "github.com/vllm/vllm-operator/pkg/api/v1"
)

const (
	defaultImage = "ghcr.io/vllm/vllm:latest"
)

// VLLMEngineReconciler reconciles a VLLMEngine object
type VLLMEngineReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

//+kubebuilder:rbac:groups=vllm.io,resources=vllmengines,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=vllm.io,resources=vllmengines/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=services,verbs=get;list;watch;create;update;patch;delete

func (r *VLLMEngineReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)

	log.Info("Reconciling VLLMEngine")

	// Fetch the VLLMEngine instance
	engine := &v1alpha1.VLLMEngine{}
	err := r.Get(ctx, req.NamespacedName, engine)
	if err != nil {
		if apierrors.IsNotFound(err) {
			log.Info("VLLMEngine resource not found")
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// Handle deletion
	if !engine.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, nil
	}

	// Default replicas
	if engine.Spec.Replicas == nil {
		replicas := int32(1)
		engine.Spec.Replicas = &replicas
	}

	// Default image
	image := engine.Spec.Image
	if image == "" {
		image = defaultImage
	}

	// Ensure Deployment exists
	deployment := r.buildDeployment(engine, image)
	err = r.reconcileDeployment(ctx, engine, deployment)
	if err != nil {
		return ctrl.Result{}, err
	}

	// Ensure Service exists
	service := r.buildService(engine)
	err = r.reconcileService(ctx, engine, service)
	if err != nil {
		return ctrl.Result{}, err
	}

	// Update status
	engine.Status.Replicas = *engine.Spec.Replicas
	engine.Status.ServiceURL = fmt.Sprintf("http://%s.%s.svc.cluster.local:8000", service.Name, req.Namespace)

	// Set Ready condition
	meta.SetStatusCondition(&engine.Status.Conditions, metav1.Condition{
		Type:    "Ready",
		Status:  metav1.ConditionTrue,
		Reason:  "Reconciled",
		Message: "vLLM Engine reconciled successfully",
	})

	err = r.Status().Update(ctx, engine)
	if err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

func (r *VLLMEngineReconciler) buildDeployment(engine *v1alpha1.VLLMEngine, image string) *appsv1.Deployment {
	replicas := *engine.Spec.Replicas
	gpuCount := engine.Spec.GPUResource.Count
	if gpuCount == 0 {
		gpuCount = 1
	}

	dep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      engine.Name,
			Namespace: engine.Namespace,
			Labels:    map[string]string{"app": "vllm", "vllmengine": engine.Name},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "vllm", "vllmengine": engine.Name},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"app": "vllm", "vllmengine": engine.Name},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "vllm",
						Image: image,
						Args:  r.buildArgs(engine),
						Ports: []corev1.ContainerPort{{ContainerPort: 8000}},
						Resources: corev1.ResourceRequirements{
							Requests: corev1.ResourceList{
								corev1.ResourceCPU:    resource.MustParse("2"),
								corev1.ResourceMemory: resource.MustParse("8Gi"),
							},
						},
					}},
				},
			},
		},
	}

	// Add GPU resource if requested
	if gpuCount > 0 {
		dep.Spec.Template.Spec.Containers[0].Resources.Requests["nvidia.com/gpu"] = resource.MustParse(fmt.Sprintf("%d", gpuCount))
		dep.Spec.Template.Spec.NodeSelector = map[string]string{"nvidia.com/gpu.product": "NVIDIA-A100-SXM4-80GB"}
	}

	ctrl.SetControllerReference(engine, dep, r.Scheme)
	return dep
}

func (r *VLLMEngineReconciler) buildArgs(engine *v1alpha1.VLLMEngine) []string {
	args := []string{
		"--model",
		engine.Spec.Model,
		"--trust-remote-code",
	}

	if engine.Spec.MaxModelLen != nil {
		args = append(args, "--max-model-len", fmt.Sprintf("%d", *engine.Spec.MaxModelLen))
	}

	if engine.Spec.TensorParallelSize != nil {
		args = append(args, "--tensor-parallel-size", fmt.Sprintf("%d", *engine.Spec.TensorParallelSize))
	}

	args = append(args, engine.Spec.Args...)

	return args
}

func (r *VLLMEngineReconciler) buildService(engine *v1alpha1.VLLMEngine) *corev1.Service {
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      engine.Name,
			Namespace: engine.Namespace,
			Labels:    map[string]string{"app": "vllm", "vllmengine": engine.Name},
		},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{"app": "vllm", "vllmengine": engine.Name},
			Ports: []corev1.ServicePort{{
				Name:     "http",
				Port:     8000,
				Protocol: corev1.ProtocolTCP,
			}},
		},
	}

	ctrl.SetControllerReference(engine, svc, r.Scheme)
	return svc
}

func (r *VLLMEngineReconciler) reconcileDeployment(ctx context.Context, engine *v1alpha1.VLLMEngine, desired *appsv1.Deployment) error {
	found := &appsv1.Deployment{}
	err := r.Get(ctx, types.NamespacedName{Name: desired.Name, Namespace: desired.Namespace}, found)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return r.Create(ctx, desired)
		}
		return err
	}

	// Update if spec changed
	if !reflect.DeepEqual(found.Spec, desired.Spec) {
		found.Spec = desired.Spec
		return r.Update(ctx, found)
	}

	return nil
}

func (r *VLLMEngineReconciler) reconcileService(ctx context.Context, engine *v1alpha1.VLLMEngine, desired *corev1.Service) error {
	found := &corev1.Service{}
	err := r.Get(ctx, types.NamespacedName{Name: desired.Name, Namespace: desired.Namespace}, found)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return r.Create(ctx, desired)
		}
		return err
	}

	// Update ports if needed
	if !equality.Semantic.DeepEqual(found.Spec.Ports, desired.Spec.Ports) {
		found.Spec.Ports = desired.Spec.Ports
		return r.Update(ctx, found)
	}

	return nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *VLLMEngineReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.VLLMEngine{}).
		Owns(&appsv1.Deployment{}).
		Owns(&corev1.Service{}).
		Complete(r)
}
