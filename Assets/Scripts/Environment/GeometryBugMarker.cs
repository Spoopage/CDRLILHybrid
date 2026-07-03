using UnityEngine;

public class GeometryBugMarker : MonoBehaviour
{
    [Header("Settings")]
    public float detectionRadius = 1.5f;
    public LayerMask groundLayer;

    [Header("Debug")]
    public bool showGizmo = true;

    private Collider[] affectedColliders;

    void Start()
    {
        ApplyBug();
    }

    public void ApplyBug()
    {
        affectedColliders = Physics.OverlapSphere(
            transform.position,
            detectionRadius,
            groundLayer
        );

        foreach (Collider col in affectedColliders)
        {
            col.enabled = false;
            Debug.Log($"Geometry Bug aktif: collider {col.gameObject.name} " +
                     $"dinonaktifkan di {transform.position}");
        }
    }

    public void RemoveBug()
    {
        if (affectedColliders == null) return;
        foreach (Collider col in affectedColliders)
        {
            if (col != null) col.enabled = true;
        }
    }

    private void OnDrawGizmos()
    {
        if (!showGizmo) return;
        Gizmos.color = new Color(1f, 0f, 0f, 0.4f);
        Gizmos.DrawSphere(transform.position, detectionRadius);
        Gizmos.DrawWireSphere(transform.position, detectionRadius);
    }
}
