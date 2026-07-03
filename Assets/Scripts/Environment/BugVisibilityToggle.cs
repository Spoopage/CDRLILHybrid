using UnityEngine;

public class BugVisibilityToggle : MonoBehaviour
{
    [ContextMenu("Sembunyikan Semua Bug Mesh (Mode Inference)")]
    public void HideAllBugMeshes()
    {
        foreach (GameObject obj in GameObject.FindGameObjectsWithTag("GroundTruthBug"))
        {
            MeshRenderer mr = obj.GetComponent<MeshRenderer>();
            if (mr != null) mr.enabled = false;
        }
        Debug.Log("Semua bug mesh disembunyikan. Collider tetap aktif.");
    }

    [ContextMenu("Tampilkan Semua Bug Mesh (Mode Debug)")]
    public void ShowAllBugMeshes()
    {
        foreach (GameObject obj in GameObject.FindGameObjectsWithTag("GroundTruthBug"))
        {
            MeshRenderer mr = obj.GetComponent<MeshRenderer>();
            if (mr != null) mr.enabled = true;
        }
        Debug.Log("Semua bug mesh ditampilkan.");
    }
}