using UnityEngine;

public class PhysicsBugTrigger : MonoBehaviour
{
    public float explosionForce = 400f;

    private void OnTriggerEnter(Collider other)
    {
        Rigidbody rb = other.GetComponent<Rigidbody>();
        if (rb != null)
        {
            Vector3 randomDir = new Vector3(
                Random.Range(-1f, 1f),
                Random.Range(0.5f, 1f),
                Random.Range(-1f, 1f)
            ).normalized;
            rb.AddForce(randomDir * explosionForce, ForceMode.Impulse);
        }
    }
}